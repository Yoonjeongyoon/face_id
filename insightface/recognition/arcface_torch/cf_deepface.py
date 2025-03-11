#!/usr/bin/env python3
"""
Facenet-PyTorch를 이용한 LFW 평가 스크립트
------------------------------------------
이 스크립트는 LFW(Labeled Faces in the Wild) 데이터셋을 대상으로 얼굴 인식 성능을 평가합니다.
전체 파이프라인은 다음과 같습니다:
  1. 원본 이미지에서 얼굴을 검출하고 정렬한 후, MTCNN을 통해 얼굴을 크롭하여 저장합니다.
  2. 크롭된 이미지를 불러와서 InceptionResnetV1(사전 학습된 모델)을 통해 임베딩을 추출합니다.
  3. pairs.txt 파일에 정의된 이미지 쌍에 대해 임베딩 간의 거리(유클리드 혹은 코사인 기반)를 계산하고
     ROC, Accuracy 등 평가 지표를 산출합니다.

Notebook에서 사용된 각 셀을 하나의 .py 파일로 변환한 예시입니다.
"""

# %% 1. 라이브러리 임포트 및 설정
import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms

# facenet-pytorch 라이브러리에서 얼굴 검출 및 임베딩 모델 임포트
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face

# 평가에 필요한 추가 라이브러리
from sklearn.model_selection import KFold
from scipy import interpolate

# %% 2. 데이터 및 하이퍼파라미터 설정
data_dir = 'data/lfw/lfw'  # LFW 원본 이미지 폴더
pairs_path = 'data/lfw/pairs.txt'  # LFW 평가를 위한 이미지 쌍 정보 파일

batch_size = 16
epochs = 15
# Windows에서는 num_workers=0, 그 외 운영체제에서는 8로 설정
workers = 0 if os.name == 'nt' else 8

# %% 3. 디바이스 설정 (GPU 사용 가능 시 GPU 할당)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# %% 4. MTCNN 객체 생성 (얼굴 검출 및 정렬)
# image_size: 출력 이미지 크기, margin: 얼굴 주변 마진, selection_method: 여러 얼굴 중 중심에 가까운 얼굴 선택
mtcnn = MTCNN(
    image_size=160,
    margin=14,
    device=device,
    selection_method='center_weighted_size'
)

# %% 5. 원본 이미지 데이터셋 로드
# ImageFolder를 사용하여 data_dir 내의 폴더 구조에 따라 이미지를 로드합니다.
orig_img_ds = datasets.ImageFolder(data_dir, transform=None)

# %% 6. 데이터셋의 class label을 이미지 경로로 덮어쓰기
# 이후 크롭된 이미지 저장 및 매칭을 위해 원본 경로가 label로 사용됩니다.
orig_img_ds.samples = [
    (p, p)
    for p, _ in orig_img_ds.samples
]

# DataLoader 생성: training.collate_pil을 collate_fn으로 사용하여 PIL 이미지 배치 생성
loader = DataLoader(
    orig_img_ds,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

# %% 7. MTCNN을 이용하여 얼굴 크롭 수행
crop_paths = []  # 크롭된 이미지 경로를 저장할 리스트
box_probs = []  # (선택사항) 얼굴 검출 확률 저장 (후에 사용하지 않을 수도 있음)
for i, (x, b_paths) in enumerate(loader):
    # 원본 경로에서 'data/lfw/lfw' 부분을 'data/lfw/lfw_cropped'로 변경하여 저장 경로 결정
    crops = [p.replace(data_dir, data_dir + '_cropped') for p in b_paths]
    # MTCNN이 입력 배치 x에서 얼굴을 검출하고 크롭된 이미지를 지정된 경로에 저장
    mtcnn(x, save_path=crops)
    crop_paths.extend(crops)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

# %% 8. MTCNN 객체 제거 및 GPU 메모리 정리
del mtcnn
torch.cuda.empty_cache()

# %% 9. 크롭된 이미지 데이터셋 및 임베딩 추출을 위한 DataLoader 생성
# 전처리 파이프라인: numpy float32 변환, 텐서 변환, 그리고 이미지 표준화(fixed_image_standardization)
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

# 크롭된 이미지 폴더에서 ImageFolder 데이터셋 생성
dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)

# 순차적으로 데이터를 불러오기 위해 SequentialSampler 사용
embed_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SequentialSampler(dataset)
)

# %% 10. 사전 학습된 InceptionResnetV1 모델 로드 (임베딩 추출용)
resnet = InceptionResnetV1(
    classify=False,
    pretrained='vggface2'
).to(device)

# %% 11. 크롭된 이미지에서 얼굴 임베딩 추출
classes = []  # 이미지의 클래스(폴더명)를 저장 (평가 시 사용)
embeddings = []  # 각 이미지에 대한 임베딩 벡터 저장
resnet.eval()  # 평가 모드로 전환 (dropout 등 비활성화)
with torch.no_grad():
    for xb, yb in embed_loader:
        xb = xb.to(device)
        b_embeddings = resnet(xb)  # 모델을 통해 임베딩 벡터 추출
        b_embeddings = b_embeddings.to('cpu').numpy()  # CPU로 옮긴 후 numpy array로 변환
        classes.extend(yb.numpy())
        embeddings.extend(b_embeddings)

# %% 12. 경로와 임베딩을 매핑하는 딕셔너리 생성
embeddings_dict = dict(zip(crop_paths, embeddings))


# %% 13. LFW 평가를 위한 함수 정의
# 아래 함수들은 LFW 데이터셋 평가 프로토콜에 맞춰 임베딩 간의 거리 계산, ROC 및 Accuracy 등을 계산합니다.
def distance(embeddings1, embeddings2, distance_metric=0):
    """
    두 임베딩 벡터 간의 거리를 계산합니다.
    distance_metric:
      0 -> 유클리드 거리
      1 -> 코사인 유사도 기반 거리
    """
    if distance_metric == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), axis=1)
    elif distance_metric == 1:
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise Exception('Undefined distance metric %d' % distance_metric)
    return dist


def calculate_accuracy(threshold, dist, actual_issame):
    """
    주어진 임계값(threshold)에서 예측한 동일 인물 여부와 실제 라벨을 비교하여
    true positive, false positive, accuracy 및 오류 지표를 계산합니다.
    """
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc, is_fp, is_fn


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    """
    K-Fold 교차 검증을 통해 ROC 커브(민감도, 특이도)와 Accuracy를 계산합니다.
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive = []
    is_false_negative = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # 훈련 세트에서 최적 임계값 탐색
        acc_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _, _ = calculate_accuracy(threshold, dist[train_set],
                                                                      actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, _, _ = calculate_accuracy(threshold,
                                                                                                       dist[test_set],
                                                                                                       actual_issame[
                                                                                                           test_set])
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                                    actual_issame[test_set])

        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)

    tpr = np.mean(tprs, axis=0)
    fpr = np.mean(fprs, axis=0)
    return tpr, fpr, accuracy, is_false_positive, is_false_negative


def calculate_val_far(threshold, dist, actual_issame):
    """
    주어진 임계값에서의 검증율(val)과 허용 오인율(FAR)을 계산합니다.
    """
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
    """
    목표 FAR에 해당하는 임계값을 찾아 검증율을 계산합니다.
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # 훈련 세트에서 목표 FAR에 해당하는 임계값 찾기
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    """
    LFW 평가를 위해 임베딩을 두 그룹(각 쌍의 이미지)으로 나누고,
    ROC 및 검증 지표를 계산합니다.
    반환 값:
      - tpr: true positive rate 배열
      - fpr: false positive rate 배열
      - accuracy: 각 fold별 Accuracy
      - val: 검증율 (validation rate)
      - val_std: 검증율의 표준편차
      - far: 평균 허용 오인율 (false acceptance rate)
      - fp, fn: false positive와 false negative 배열
    """
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn = calculate_roc(thresholds, embeddings1, embeddings2,
                                               np.asarray(actual_issame), nrof_folds=nrof_folds,
                                               distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds,
                                      distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far, fp, fn


def add_extension(path):
    """
    주어진 파일 경로에 .jpg 또는 .png 확장자를 추가하여 해당 파일이 존재하면 반환합니다.
    """
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "{}" with extension png or jpg.'.format(path))


def get_paths(lfw_dir, pairs):
    """
    pairs 리스트를 순회하며 이미지 쌍의 경로와, 동일 인물 여부(issame)를 결정합니다.
    두 이미지 모두 존재할 때만 리스트에 추가합니다.
    """
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            # 같은 사람인 경우: [이름, index1, index2]
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            # 다른 사람인 경우: [이름1, index1, 이름2, index2]
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # 두 이미지 모두 존재할 경우에만 추가
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped {} image pairs'.format(nrof_skipped_pairs))
    return path_list, issame_list


def read_pairs(pairs_filename):
    """
    pairs.txt 파일을 읽어 이미지 쌍 정보를 numpy 배열로 반환합니다.
    첫 번째 라인은 헤더이므로 건너뜁니다.
    """
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)


# %% 14. LFW 평가 수행
# 1) pairs.txt 파일을 읽어 pairs 배열 생성
pairs = read_pairs(pairs_path)
# 2) 크롭된 이미지 경로와 동일 인물 여부 리스트를 생성
path_list, issame_list = get_paths(data_dir + '_cropped', pairs)
# 3) pairs에 해당하는 이미지의 임베딩을 배열로 생성 (embeddings_dict에서 조회)
embeddings_eval = np.array([embeddings_dict[path] for path in path_list])

# 4) evaluate 함수를 통해 ROC, Accuracy, 검증율 등 평가 수행
tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings_eval, issame_list)

# %% 15. 평가 결과 출력
print("Accuracy for each fold:", accuracy)
print("Mean Accuracy:", np.mean(accuracy))

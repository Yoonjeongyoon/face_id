import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch.cuda
from sklearn.metrics import roc_curve, auc
import numpy as np
import math
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import argparse
from sklearn.model_selection import KFold
from scipy import interpolate
import sys
import warnings
from backbones import get_model

warnings.filterwarnings(("ignore"))
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

parser = argparse.ArgumentParser(description='do ijb test')
parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--image-path', default='', type=str, help='')
parser.add_argument('--result-dir', default='.', type=str, help='')
parser.add_argument('--batch-size', default=16, type=int, help='')
parser.add_argument('--network', default='iresnet50', type=str, help='')
args = parser.parse_args()
# 이미지 폴더 경로

dataset_dir = args.image_path
model_path = args.model_prefix

batch_size = args.batch_size
device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

trans = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])
class FaceDataset(Dataset):
    def __init__(self, path_list, transform=None):
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)
    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, img_path
weight = torch.load(model_path)
resnet = get_model(args.network, dropout=0, fp16=True).cuda()
resnet.load_state_dict(weight)
model = torch.nn.DataParallel(resnet)
model.eval()
def get_embeddings_from_pathlist(path_list, batch_size=16):
    dataset = FaceDataset(path_list, transform=trans)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings_dict = {}
    with torch.no_grad():
        for batch_imgs, batch_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            batch_embeds = model(batch_imgs)  # 배치 단위로 임베딩 계산
            batch_embeds = batch_embeds.cpu().numpy()  # CPU로 변환 후 numpy 배열로 변환

            for path, embed in zip(batch_paths, batch_embeds):
                embeddings_dict[path] = embed

    return embeddings_dict


# 임베딩 거리를 계산하는 함수 (코사인 유사도)
def distance(embeding1, embeding2, distance_metric = 0):
    if distance_metric ==0:
        dot = np.sum(np.multiply(embeding1,embeding2), axis=1)# 벡터의 내적
        norm = np.linalg.norm(embeding1, ord=2, axis=1) * np.linalg.norm(embeding2,ord=2, axis=1)# 각 벡터의 절대 값의 곱
        cos_similarity = dot / norm
        cos_similarity = np.clip(cos_similarity, -1.0, 1.0)
        dist = np.arccos(cos_similarity) / math.pi # 코사인 유사도 기반 각도를 0~1 사의로 정규화 = 아크코사인 사용
    else:
        raise Exception("Undefined distance metirc %d" % distance_metric)
    return dist

def get_paths(dataset_dir):
    nrof_skipped_pairs =0
    path_list = []
    issame_list = []

    genuine_path = os.path.join(dataset_dir, "gen") # genuine set의 경로
    genuine_folders = sorted(os.listdir(genuine_path))# genuine 폴더의 리스트
    genuine_count = len(genuine_folders) #총 폴더 수
    genuine_split = np.array_split(genuine_folders, 10)

    imposter_path = os.path.join(dataset_dir, "imp")
    imposter_folders = sorted(os.listdir(imposter_path))
    imposter_count = len(imposter_folders)
    imposter_split = np.array_split(imposter_folders, 10)
# 10 fold를 위해 각 fold에 genuine과 imposter를 10분에 1씩 할당
    for fold_idx in range(10):
        for folder in genuine_split[fold_idx]:
            folder_path = os.path.join(genuine_path, folder)
            images = sorted(os.listdir(folder_path))
            if len(images)==2:
                path0 = os.path.join(folder_path,  images[0])
                path1 = os.path.join(folder_path, images[1])
                path_list += [path0, path1]
                issame_list.append(1)
            else:
                nrof_skipped_pairs +=1

        for folder in imposter_split[fold_idx]:
            folder_path = os.path.join(imposter_path, folder)
            images = sorted(os.listdir(folder_path))
            if len(images)==2:
                path0 = os.path.join(folder_path, images[0])
                path1 = os.path.join(folder_path, images[1])
                path_list += [path0, path1]
                issame_list.append(0)
            else:
                nrof_skipped_pairs +=1
    if nrof_skipped_pairs > 0:
        print(f'Skipped {nrof_skipped_pairs} image paiars')
    return path_list, issame_list

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

"""
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0,
                  subtract_mean=False):
   
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
"""
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

        # far_train에서 중복 제거
        unique_far_train, unique_indices = np.unique(far_train, return_index=True)

        if len(unique_far_train) > 1 and np.max(unique_far_train) >= far_target:
            unique_thresholds = thresholds[unique_indices]  # 중복 제거된 thresholds 선택
            f = interpolate.interp1d(unique_far_train, unique_thresholds, kind='slinear', fill_value="extrapolate")
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
    thresholds = np.arange(0, 1, 0.001)
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




path_list, issame_list = get_paths(dataset_dir)
embeddings_dict=get_embeddings_from_pathlist(path_list, batch_size=16)
embeddings_eval = np.array([embeddings_dict[path] for path in path_list])
tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings_eval, issame_list)
print("Accuracy for each fold:", accuracy)
print("Mean Accuracy:", np.mean(accuracy))
print("현재 작업 디렉토리:", os.getcwd())
import matplotlib.pyplot as plt

# ROC Curve 그리기
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc(fpr, tpr):.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 대각선 기준선
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
# 먼저, embeddings_eval에서 쌍별로 임베딩을 분리합니다.
embeddings1 = embeddings_eval[0::2]
embeddings2 = embeddings_eval[1::2]

# 전체 쌍에 대해 거리를 계산
dists = distance(embeddings1, embeddings2, distance_metric=0)

# genuine 쌍(issame_list가 1인 경우)만 선택
issame_array = np.array(issame_list)
genuine_dists = dists[issame_array == 1]
imposter_dists = dists[issame_array == 0]
# genuine 쌍의 평균 거리를 계산
average_genuine_dist = np.mean(genuine_dists)
print("Average distance for genuine pairs:", average_genuine_dist)
std_genuine_dists = np.std(genuine_dists)
print("Standard deviation for genuine pairs:", std_genuine_dists)

average_imposter_dist = np.mean(imposter_dists)
print("Average distance for imposter pairs:", average_imposter_dist)
std_imposter_dists = np.std(imposter_dists)
print("Standard deviation for imposter pairs:", std_imposter_dists)

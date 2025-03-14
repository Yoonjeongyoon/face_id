from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import os.path as osp
import numba
import tqdm
import torch.distributed as dist

from fr.utils.dataset import dataset_dict

checkpoint_path = "mtlface_checkpoints.tar"
@torch.no_grad()
def generate_embeddings(BACKBONE, dataset_name, batch_size, image_size=112):
    BACKBONE.eval()
    transform = transforms.Compose([
        # transforms.Resize([128, 128]),  # smaller side resized
        # transforms.CenterCrop([112, 112]),
        transforms.Resize([image_size, image_size]),  # smaller side resized
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    test_dataset = dataset_dict[dataset_name](transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
        num_workers=16, drop_last=False
    )
    embeddings = []
    labels = []
    disable = False
    if dist.is_initialized():
        if dist.get_rank() != 0:
            disable = True
    for images, _labels in tqdm.tqdm(test_loader, disable=disable):
        images = images.cuda()
        embedding = BACKBONE(images) + BACKBONE(torch.flip(images, dims=(3,)))
        # embedding = BACKBONE(inputs)
        # embedding = torch.cat([BACKBONE(inputs) + BACKBONE(torch.flip(inputs, dims=(3, )))])
        embedding = F.normalize(embedding)
        embeddings.append(embedding.cpu().numpy())
        labels.append(_labels.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels.astype(int)


def calculate_accuracy(threshold, dist, is_same):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, is_same))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(is_same)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(is_same)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), is_same))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(embeddings1, embeddings2, is_same, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    thresholds = np.arange(0, 4, 0.01)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    #     dist = 1 - np.sum(embeddings1 * embeddings2, 1)
    #     thresholds = np.arange(0.0, 1.0, 0.00025)
    #     thresholds = np.sort(dist)

    nrof_pairs = min(len(is_same), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], is_same[train_set])
        best_threshold_index = np.argmax(acc_train)
        #         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 is_same[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      is_same[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    auc = metrics.auc(fpr, tpr)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc,
        'accuracy': accuracy.mean(),
        'best_threshold': best_thresholds.mean()
    }


def general_evalute(BACKBONE, dataset_name, batch_size):
    '''

    :param BACKBONE: trained BACKBONE model
    :param dataset_name: eval dataset name, including
        * lfw,
        * cacd_vs,
        * calfw,
    :param batch_size: test loader batch size;
    :return: AUC and ACC
    '''
    embeddings = generate_embeddings(BACKBONE, dataset_name, batch_size)
    pairs = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', dataset_name, 'pairs.txt')
    pairs = np.loadtxt(pairs).astype(int)
    embeddings1, embeddings2 = embeddings[pairs[:, 0], :], embeddings[pairs[:, 1], :]
    is_same = pairs[:, 2].astype(bool)
    return calculate_roc(embeddings1, embeddings2, is_same, nrof_folds=10)
# 체크포인트 파일 경로
checkpoint_path = "mtlface_checkpoints.tar"

# 모델 생성 (입력 크기 및 특징 차원은 실제 모델에 맞게 설정)
BACKBONE = Backbone(input_size=112, num_features=512)  # 모델 구조 확인 후 수정

# 체크포인트 로드
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 모델 가중치 적용
BACKBONE.load_state_dict(checkpoint['model_state_dict'])
BACKBONE = BACKBONE.cuda()  # GPU 사용
BACKBONE.eval()  # 평가 모드로 설정

# 데이터셋 및 배치 크기 설정
dataset_name = "lfw"  # 사용할 데이터셋 이름 (lfw, cacd_vs, calfw 등)
batch_size = 64  # 적절한 배치 크기 설정

# 얼굴 임베딩 생성
embeddings, labels = generate_embeddings(BACKBONE, dataset_name, batch_size)

print("✅ 임베딩 생성 완료!")
print("임베딩 shape:", embeddings.shape)
print("레이블 개수:", labels.shape)

# 모델 검증 수행
evaluation_result = general_evalute(BACKBONE, dataset_name, batch_size)

# 결과 출력
print("✅ 검증 완료!")
print("AUC:", evaluation_result['auc'])
print("Accuracy:", evaluation_result['accuracy'])
print("Best Threshold:", evaluation_result['best_threshold'])
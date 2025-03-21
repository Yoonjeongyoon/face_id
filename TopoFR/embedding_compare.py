import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from backbones import get_model
from utils.utils_config import get_config

# 이미지 전처리 설정
trans = transforms.Compose([
    transforms.Resize((112, 112)), #모델 입력 사이즈 112*112
    transforms.ToTensor()
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model_path = 'work_dirs/MS1MV2_R100_TopoFR_9695.pt'
network = 'r100'
cfg = get_config('configs/ms1mv2_r100.py')
weight = torch.load(model_path)
resnet = get_model(network, dropout=0, fp16=False, num_features=cfg.embedding_size, num_classes=cfg.num_classes).to(device)
resnet.load_state_dict(weight)
model = torch.nn.DataParallel(resnet)
model.eval()

# 이미지로부터 임베딩을 추출하는 함수
def get_embedding(img_path):
    image = Image.open(img_path).convert('RGB')
    image = trans(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image, phase='infer').cpu().numpy().flatten()

    return embedding

# 두 임베딩 간의 코사인 유사도 기반 거리 계산 함수
def calculate_distance(embed1, embed2):
    eps = 1e-10
    dot = np.dot(embed1, embed2)
    norm = np.linalg.norm(embed1) * np.linalg.norm(embed2) + eps
    cos_similarity = dot / norm
    distance = np.arccos(cos_similarity) / np.pi
    return distance


if __name__ == '__main__':
    img_path1 = 'testset1/pair1_5_KimMin-joung_71_w.jpg'
    img_path2 = 'testset1/pair2_6_KimMin-joung_68_w.jpg'

    embedding1 = get_embedding(img_path1)
    embedding2 = get_embedding(img_path2)

    dist = calculate_distance(embedding1, embedding2)

    print(f'두 이미지의 임베딩 벡터 거리: {dist:.6f}')

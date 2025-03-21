import os
import numpy as np
from datasets import Dataset
from functools import partial
from PIL import Image
import argparse


def entry_for_row(index, image, is_same):
    """ 하나의 샘플을 딕셔너리 형태로 변환 """
    return {
        "image": image,
        "index": index,
        "is_same": is_same
    }


def generate_entries(image_list, is_same_list):
    """ 이미지와 is_same 정보를 바탕으로 데이터 생성 """
    for index, (image, is_same) in enumerate(zip(image_list, is_same_list)):
        yield entry_for_row(index, image, is_same)


def load_images_from_folders(dataset_dir):
    """ gen과 imp 폴더에서 이미지를 불러와서 리스트로 변환 """
    image_list = []
    is_same_list = []

    for label, folder in enumerate(["gen", "imp"]):  # gen = True(1), imp = False(0)
        folder_path = os.path.join(dataset_dir, folder)
        for pair_id in sorted(os.listdir(folder_path)):  # 0000 ~ 1999
            pair_folder = os.path.join(folder_path, pair_id)
            images = sorted(os.listdir(pair_folder))  # 2장 이미지
            if len(images) == 2:
                img1 = Image.open(os.path.join(pair_folder, images[0])).convert("RGB")
                img2 = Image.open(os.path.join(pair_folder, images[1])).convert("RGB")
                image_list.append(img1)
                image_list.append(img2)
                is_same_list.append(bool(label))  # gen: True, imp: False

    return image_list, is_same_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CACD_VS to Hugging Face datasets')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to CACD_VS dataset')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save Hugging Face dataset')
    args = parser.parse_args()

    # 이미지 로드
    pil_images, issame_list = load_images_from_folders(args.dataset_dir)

    # is_same 리스트를 두 배로 확장
    repeated_issame_list = np.stack([issame_list, issame_list], axis=0).transpose().flatten().tolist()
    assert repeated_issame_list[::2] == issame_list
    assert not repeated_issame_list[:len(issame_list)] == issame_list
    assert len(repeated_issame_list) == len(pil_images)

    # 데이터셋 변환
    generator = partial(generate_entries, pil_images, repeated_issame_list)
    ds = Dataset.from_generator(generator)

    # 저장
    dataset_name = "Asian_older_huggingface"
    os.makedirs(args.save_dir, exist_ok=True)
    ds.save_to_disk(os.path.join(args.save_dir, dataset_name), num_shards=1)

    print(f"✅ Hugging Face dataset 저장 완료: {os.path.join(args.save_dir, dataset_name)}")

    # 샘플 저장
    os.makedirs(os.path.join(args.save_dir, dataset_name, 'examples'), exist_ok=True)
    for i in range(5):
        ds[i]['image'].save(os.path.join(args.save_dir, dataset_name, 'examples', f'{i}.jpg'))

    print(f"✅ 샘플 5개 저장 완료: {os.path.join(args.save_dir, dataset_name, 'examples')}")

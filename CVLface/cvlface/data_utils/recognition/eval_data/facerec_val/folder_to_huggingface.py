import os
import shutil
import argparse
import numpy as np
from datasets import Dataset, concatenate_datasets
from PIL import Image


def entry_for_row(index, image, is_same):
    return {
        "image": image,
        "index": index,
        "is_same": is_same
    }


def load_images_from_folders(dataset_dir):
    image_list = []
    is_same_list = []

    for label, folder in enumerate(["gen", "imp"]):
        is_same = folder == "gen"
        folder_path = os.path.join(dataset_dir, folder)
        pair_ids = sorted(os.listdir(folder_path), key=lambda x: int(x))

        for pair_id in pair_ids:
            pair_folder = os.path.join(folder_path, pair_id)
            images = sorted(os.listdir(pair_folder))
            if len(images) == 2:
                img1 = Image.open(os.path.join(pair_folder, images[0])).convert("RGB")
                img2 = Image.open(os.path.join(pair_folder, images[1])).convert("RGB")
                image_list.append(img1)
                image_list.append(img2)
                is_same_list.append(is_same)  # gen: True, imp: False

    return image_list, is_same_list


def split_into_folds(src_dir, dst_base_dir, num_folds=10):
    for label in ['gen', 'imp']:
        folder_path = os.path.join(src_dir, label)
        pair_ids = sorted(os.listdir(folder_path), key=lambda x: int(x))
        fold_size = len(pair_ids) // num_folds

        for fold_idx in range(num_folds):
            fold_label_path = os.path.join(dst_base_dir, f"fold_{fold_idx}", label)
            os.makedirs(fold_label_path, exist_ok=True)

            start_idx = fold_idx * fold_size
            end_idx = (fold_idx + 1) * fold_size
            selected = pair_ids[start_idx:end_idx]

            for pair_id in selected:
                src = os.path.join(folder_path, pair_id)
                dst = os.path.join(fold_label_path, pair_id)
                shutil.copytree(src, dst)

    print(f"{num_folds} folds로 나누기 완료: {dst_base_dir}")


def convert_fold_to_dataset(fold_path, global_start_index=0):
    pil_images, issame_list = load_images_from_folders(fold_path)
    repeated_issame_list = np.stack([issame_list, issame_list], axis=0).transpose().flatten().tolist()
    assert repeated_issame_list[::2] == issame_list
    assert len(repeated_issame_list) == len(pil_images)

    entries = []
    for i, (image, is_same) in enumerate(zip(pil_images, repeated_issame_list)):
        entries.append({
            "image": image,
            "index": global_start_index + i,
            "is_same": is_same
        })

    ds = Dataset.from_list(entries)
    return ds, global_start_index + len(pil_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dataset_dir', type=str, required=True, help='원본 gen/imp 폴더 위치')
    parser.add_argument('--output_base_dir', type=str, required=True, help='Hugging Face dataset 저장 위치')
    args = parser.parse_args()

    folded_dataset_dir = os.path.join(args.output_base_dir, "folded_dataset")
    merged_output_dir = os.path.join(args.output_base_dir, "merged_dataset")

    # Step 1: fold 나누기
    split_into_folds(args.src_dataset_dir, folded_dataset_dir, num_folds=10)

    # Step 2: fold들 순서대로 dataset으로 변환 + 전역 index 부여
    all_folds = []
    current_index = 0
    for i in range(10):
        fold_path = os.path.join(folded_dataset_dir, f"fold_{i}")
        ds, current_index = convert_fold_to_dataset(fold_path, global_start_index=current_index)
        all_folds.append(ds)

    # Step 3: 병합
    merged_dataset = concatenate_datasets(all_folds)

    # Step 4: 저장
    os.makedirs(merged_output_dir, exist_ok=True)
    merged_dataset.save_to_disk(merged_output_dir, num_shards=1)
    print(f"전체 fold 병합 Hugging Face dataset 저장 완료: {merged_output_dir}")

    # Step 5: 샘플 이미지 저장
    sample_dir = os.path.join(merged_output_dir, 'examples')
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(5):
        merged_dataset[i]['image'].save(os.path.join(sample_dir, f'{i}.jpg'))
    print(f"샘플 이미지 저장 완료: {sample_dir}")

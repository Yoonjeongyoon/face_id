import os
from pathlib import Path

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))

def find_leaf_folders_with_few_images(root_path):
    root_path = Path(root_path)

    for dirpath, dirnames, filenames in os.walk(root_path):
        # dirpath는 현재 폴더 경로
        # dirnames가 비어 있으면, 하위 폴더가 없는 말단 폴더
        if not dirnames:
            image_count = sum(1 for f in filenames if is_image_file(f))
            if image_count < 2:
                print(f"❌ 이미지 부족: {dirpath}")

# 예시 경로
find_leaf_folders_with_few_images("LFW")

def find_leaf_folders_with_few_images(root_path):
    root_path = Path(root_path)

    for dirpath, dirnames, filenames in os.walk(root_path):
        # dirpath는 현재 폴더 경로
        # dirnames가 비어 있으면, 하위 폴더가 없는 말단 폴더
        if not dirnames:
            image_count = sum(1 for f in filenames if is_image_file(f))
            if image_count < 2:
                print(f"❌ 이미지 부족: {dirpath}")

# 예시 경로
find_leaf_folders_with_few_images("/path/to/root")

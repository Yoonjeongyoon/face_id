import os
import subprocess

# 이미지 폴더들이 들어 있는 상위 디렉토리 경로
base_dataset_dir = 'work_dirs/corruptions/AgeDB_30/test'  # 또는 다른 경로


# 이미지 폴더들 순회
for folder_name in sorted(os.listdir(base_dataset_dir)):
    image_path = os.path.join(base_dataset_dir, folder_name)
    if not os.path.isdir(image_path):
        continue  # 디렉토리가 아니면 스킵

    # 실행 명령 구성
    cmd = [
        "python", "eval_lfw_tent_benchmark.py",  # 여기에 원래 실행할 .py 파일 이름
        "--image_path", image_path,
    ]

    # 실행
    subprocess.run(cmd)

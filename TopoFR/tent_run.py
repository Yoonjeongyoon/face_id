import os
import subprocess

# 이미지 폴더들이 들어 있는 상위 디렉토리 경로
base_dataset_dir = 'work_dirs/corruptions/LFW/test' 


# 이미지 폴더들 순회
for folder_name in sorted(os.listdir(base_dataset_dir)):
    image_path = os.path.join(base_dataset_dir, folder_name)
    if not os.path.isdir(image_path):
        continue 

    # 실행 명령 구성
    cmd = [
        "python", "eval_lfw_tent_benchmark.py", 
        "--image_path", image_path,
    ]

    # 실행
    subprocess.run(cmd)

import os
import subprocess

# 이미지 폴더들이 들어 있는 상위 디렉토리 경로
base_dataset_dir = 'work_dirs/corruptions/CPLFW/test'  # 또는 다른 경로

# log 저장 디렉토리
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

# 이미지 폴더들 순회
for folder_name in sorted(os.listdir(base_dataset_dir)):
    image_path = os.path.join(base_dataset_dir, folder_name)
    if not os.path.isdir(image_path):
        continue  # 디렉토리가 아니면 스킵

    # 로그 파일명 설정
    log_file = f"CPLFW_{folder_name}_benchmark_results.txt"
    log_path = os.path.join(log_dir, log_file)

    print(f"▶ Testing folder: {folder_name}")
    print(f"   → Input path : {image_path}")
    print(f"   → Output log : {log_path}")

    # 실행 명령 구성
    cmd = [
        "python", "eval_lfw_tent_benchmark.py",  # 여기에 원래 실행할 .py 파일 이름
        "--image_path", image_path,
    ]

    # 실행 환경변수 설정 (로그 저장 위치를 환경변수로 넘길 수도 있음)
    env = os.environ.copy()
    env["LOG_FILENAME"] = log_path  # 내부에서 이걸 쓰게 할 수도 있음

    # 실행
    subprocess.run(cmd, env=env)

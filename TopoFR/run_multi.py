import subprocess
import os
from multiprocessing import Pool

MODEL_PREFIX = "work_dirs/MS1MV2_R100_TopoFR_9695.pt"
BATCH_SIZE = 16
NETWORK = "r100"
BASE_PARENT_DIR = "work_dirs/corruptions"
PARENT_DIRS = [
    "Asian_celebrity_LFW_OOD"
]

def run_eval(image_path, output_dir):
    output_name = image_path.replace("/", "_").replace("\\", "_")
    output_file = os.path.join(output_dir, f"{output_name}_result.txt")

    cmd = [
        "python", "eval_age_benchmark.py",
        "--model-prefix", MODEL_PREFIX,
        "--image-path", image_path,
        "--batch-size", str(BATCH_SIZE),
        "--network", NETWORK
    ]

    print(f"Running evaluation for: {image_path}")
    try:
        with open(output_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"Completed evaluation for: {image_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred for {image_path}: {e}")

def eval_one_directory(parent_name):
    full_path = os.path.join(BASE_PARENT_DIR, parent_name)
    output_dir = f"eval_{parent_name}_results"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = [f.path for f in os.scandir(full_path) if f.is_dir()]

    print(f"\n=== 시작: {parent_name} ===")
    with Pool(4) as pool:
        pool.starmap(run_eval, [(path, output_dir) for path in image_paths])
    print(f"=== 완료: {parent_name} ===\n")

if __name__ == "__main__":
    for parent in PARENT_DIRS:
        eval_one_directory(parent)

    print("모든 작업 완료!")


import os
import sys
import torch
import torch.nn as nn
import torch.jit
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy

import tent
from backbones import get_model
from utils.utils_config import get_config

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# python topofr_tent_test.py 2>&1 | tee -a topofr_tent_test.log


class FaceDataset(Dataset):
    def __init__(self, path_list):
        self.path_list = path_list
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.path_list)
    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, img_path


class Tee:
    """콘솔과 파일에 동시에 출력하는 클래스"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()

tee_logger = Tee(os.path.join(os.path.dirname(__file__), 'log.txt'))
sys.stdout = tee_logger

def get_topofr_model(name='r50'):
    network = name
    if name == 'r50':
        model_config = 'configs/glint360k_r50.py'
        model_path = 'model/Glint360K_R50_TopoFR_9727.pt'
    elif name == 'r100':
        model_config = 'configs/ms1mv2_r100.py'
        model_path = 'work_dirs/MS1MV2_R100_TopoFR_9695.pt'
    elif name == 'r200':
        model_config = 'configs/glint360k_r200.py'
        model_path = 'model/Glint360K_R200_TopoFR_9784.pt'
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    cfg = get_config(model_config)
    model = get_model(network, dropout=0, fp16=False, num_features=cfg.embedding_size, num_classes=cfg.num_classes)
    weight = torch.load(model_path, weights_only=True)
    model.load_state_dict(weight)
    # model = torch.nn.DataParallel(model)

    model = tent.configure_model(model)
    return model

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# 0. Load Original model
models = ['r100']#['r50', 'r100', 'r200']

img_dirs = ["work_dirs/corruptions/LFW/defocus_blur_1"]#["/data/dataset/Face/benchmark/tent/calfw",
           #"/data/dataset/Face/benchmark/tent/agedb_30",
           #"/data/dataset/Face/benchmark/tent/cfp_fp",
           #"/data/dataset/Face/benchmark/tent/cplfw",
           #"/data/dataset/Face/benchmark/tent/lfw",
           #]
tent_step =  [10]#[1, 3, 5, 10, 20]
tent_batch_size = [256]#[2, 4, 8, 16, 32, 64, 128, 256]     # 2, 4, 8, 16, 32, 64, 128, 256

exp_cnt = 0
for img_dir in img_dirs:
    dataset_name = os.path.basename(img_dir)

    for model_name in models:
        if model_name == 'r200':
            tent_batch_size = [2, 4, 8, 16, 32, 64]
        else:
            tent_batch_size = [2, 4, 8, 16, 32, 64, 128, 256]

        for ts in tent_step:
            for tbs in tent_batch_size:
                print(f"Experiment {exp_cnt}: {dataset_name} - {model_name} - Tent Step: {ts} - Batch Size: {tbs}")
                experiment_name = f"{dataset_name}_{model_name}_tent_step{ts}_batch_size{tbs}"

                # Load your dataset
                img_path_list = []
                for img_file in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_file)
                    if os.path.isfile(img_path):
                        img_path_list.append(img_path)
                dataset = FaceDataset(img_path_list)
                dataloader = DataLoader(dataset, batch_size=tbs, shuffle=False)
                
                # 1. Configure your model for test-time adaptation
                model = get_topofr_model(model_name)
                # 2. Collect BatchNorm parameters for optimization
                params, param_names = tent.collect_params(model)
                # 3. Create an optimizer for the collected parameters
                optimizer = torch.optim.Adam(params, lr=1e-3)  # Example optimizer
                # 4. Create a Tent instance
                tented_model = tent.Tent(
                    model=model, 
                    optimizer=optimizer, 
                    steps=ts, 
                    episodic=False)

                # 5. Use for inference (adapts automatically)
                cnt = 0
                for inputs, _ in dataloader:
                    # Move inputs to the same device as the model
                    inputs = inputs.to(next(tented_model.parameters()).device)
                    # Forward pass through the tented model
                    # with torch.no_grad():
                    outputs = tented_model(inputs)
                    cnt += 1
                    full_cnt = cnt * tbs
                    if full_cnt % (10) == 0:
                        print(f"{experiment_name} - Processed {full_cnt} images")

                # After completing test-time adaptation, save the adapted model weights
                os.makedirs('model_out', exist_ok=True)
                adapted_model_path = f'model_out/MS1MV2_{model_name}_TopoFR_adapted_{dataset_name}_epoch{ts}_batch{tbs}.pt'
                # torch.save(tented_model.model.state_dict(), "model_out/Glint360_torch_state.pt") #adapted_model_path)
                torch.save(tented_model.model, adapted_model_path)
                # model_script = torch.jit.script(tented_model.model)
                # model_script.save(adapted_model_path)
                print(f"Adapted model saved to {adapted_model_path}")

print("All experiments completed.")
sys.stdout = tee_logger.terminal
tee_logger.file.close()


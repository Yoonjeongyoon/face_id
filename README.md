# face_id  


<details>
<summary><b>Dataset</b></summary>
<br>
<details>
<summary>face_align.ipynb</summary>

| 항목 | 내용 |
|------|------|
| **이름** | [`face_align.ipynb`](face_align.ipynb) |
| **파일 경로** | `face_id/face_align.ipynb` |
| **기능** | 얼굴 정렬(Alignment) 수행<br>`insightface`의 `FaceAnalysis` 로 얼굴 검출 후 정렬된 이미지를 저장 |
| **사용 모델** | `insightface` 내장 face detection + landmark (CPU) |
| **입력 형식** | 단일 face 이미지가 있는 `lowdata/` 폴더<br>└─ 서브폴더 포함 전체 이미지 탐색 |
| **출력 형식** | 정렬된 얼굴 이미지 (`Asian_celebrity_align/`)<br>└─ 입력 폴더와 동일한 디렉터리 구조 |
| **기능 요약** | - 이미지 내 얼굴 검출<br>- 랜드마크 기반 얼굴 정렬<br>- 이미 처리된 파일 스킵 |
</details>

<details>
<summary>gen_pairs.ipynb</summary>

| 항목 | 내용 |
|------|------|
| **이름** | [`gen_pairs.ipynb`](gen_pairs.ipynb) |
| **파일 경로** | `face_id/gen_pairs.ipynb` |
| **기능** | 동일 인물 폴더 내 모든 이미지 조합을 생성한 뒤 무작위 3,000 쌍을 `gen/` 폴더에 복사하는 **genuine 페어 생성** 코드 |
| **사용 모델** | ― |
| **입력 형식** | `face_align.ipynb` 에서 정렬된 얼굴 이미지 (`Asian_celebrity_align/…`) |
| **출력 형식** | `gen/0/`, `gen/1/` … `gen/2999/`<br>└─ 각 폴더에 `pair1_<파일명>.jpg`, `pair2_<파일명>.jpg` |
| **기능 요약** | - 이미지 ≥2장인 인물 폴더에서 모든 조합 생성<br>- `random.seed(42)` 로 섞어 3,000 쌍 선정<br>- 쌍마다 고유 인덱스 폴더 생성 후 두 이미지를 복사 |
</details>

<details>
<summary>imp_pairs.py</summary>

| 항목 | 내용 |
|------|------|
| **이름** | [`imp_pairs.py`](imp_pairs.py) |
| **파일 경로** | `face_id/imp_pairs.py` |
| **기능** | **impostor(타인) 페어 생성** 코드.<br>성별(m / w)을 기준으로 **다른 인물**‑이미지 두 장을 무작위로 골라 3,000 쌍을 `imp/` 폴더에 복사 |
| **사용 모델** | ― |
| **입력 형식** | `face_align.ipynb`에서 정렬된 얼굴 이미지 (`Asian_celebrity_align/…`) |
| **출력 형식** | `imp/0/`, `imp/1/` … `imp/2999/`<br>└─ 각 폴더에 `pair1_<파일명>.jpg`, `pair2_<파일명>.jpg` |
| **기능 요약** | - 인물 폴더명(예: `홍길동_m/…`)의 이미지 확장자로 성별 (m/w) 판별<br>- 같은 성별 그룹(남 ↔ 남, 여 ↔ 여)에서 서로 다른 인물 두 명을 무작위 선택<br>- `random.seed(42)`로 재현 가능한 3 000 쌍 생성<br>- 쌍마다 고유 인덱스 폴더를 만들고 두 이미지를 복사 |
</details>
</details>
<details>
<summary><b>CVLface</b></summary>
   <br>
   <details>
     <summary>folder_to_huggingface.py</summary>
     
   | 항목 | 내용 |
   |------|------|
   | **이름** | [`folder_to_huggingface.py`](CVLface/cvlface/data_utils/recognition/eval_data/facerec_val/folder_to_huggingface.py) |
   | **파일 경로** | `face_id/CVLface/cvlface/data_utils/recognition/eval_data/facerec_val/folder_to_huggingface.py` |
   | **기능** | `gen/`·`imp/` 페어를 LFW 프로토콜에 맞게 **10‑fold** 로 분할 후, Hugging Face `Dataset`으로 변환·병합하여 adaface를 돌릴 수 있는 형태로 변환<br> 이후 evaluations/configs의 yaml파일을 수정하여 eval.py를 실행 |
   | **사용 모델** | 
   | **입력 형식** | `gen/` & `imp/` 폴더 구조 (`pair1_*.jpg`, `pair2_*.jpg`)<br>예) `gen/0/…`, `imp/42/…` |
   | **출력 형식** | `.Arrow 포맷 + 예시이미지`examples/0.jpg` … `4.jpg` |
   | **Arguments** | `--src_dataset_dir` (원본 gen/imp 경로)<br>`--output_base_dir` (HF dataset 저장 루트) |
   | **기능 요약** | 1. `split_into_folds` → gen/imp를 10개 fold로 디렉터리 복사<br>2. 각 fold를 `Dataset.from_list`로 변환하며 **전역 인덱스** 부여<br>3. `concatenate_datasets`로 병합하여 하나의 형태로 저장<br>4. 예시 이미지 5장을 `examples/`에 저장 |
   
   </details>
 <details>
<summary>eval.py</summary>

| 항목 | 내용 |
|------|------|
| **이름** | [`eval.py`](CVLface/cvlface/research/recognition/code/run_v1/recognition/eval.py) |
| **파일 경로** | `face_id/CVLface/cvlface/research/recognition/code/run_v1/eval.py` |
| **기능** | adaface의사전 학습 얼굴 인식 모델 YAML로 읽어 자동 빌드 후, 설정된 **여러 벤치마크**(LFW, CPLFW, …)를 한 번에 돌려 결과를 출력|
| **사용 모델** | `models.get_model()` 로 로드되는 Adaface사전 학습모델 |

### 평가 스크립트 간단 사용법

```bash
python eval_lfw_tent_benchmark.py \
  --num_gpu <GPU개수> \
  --eval_config_name face_id/CVLface/cvlface/research/recognition/code/run_v1/evaluations/config/<원하는 YAML> \
  --ckpt_dir <사전학습 모델 폴더>
```

- `--num_gpu`: 사용 GPU 개수
- `--eval_config_name`: 평가 설정 YAML 파일 경로
  - 예시 YAML(필요 시 값만 수정):

    ```yaml
    eval_every_n_epochs: 1
    per_epoch_evaluations:
      lfw:
        path: facerec_val/cfp_fp #평가에 수행할 데이터셋 경로 folder_to_huggingface.py로 변환하여 사용
        evaluation_type: verification  #평가 방법 LFW=verification, ijb=ijbbc, tinyface=tinyface
        color_space: RGB
        batch_size: 32 #배치사이즈
        num_workers: 4 # 데이터셋 처리 코어 개수
    ```
- `--ckpt_dir`: `.pt`, `config.yaml`, `model.yaml`이 함께 있는 체크포인트 폴더의 경로 


</details>
   <details>
     <summary>example.ipynb</summary>
     
   | 항목 | 내용 |
   |------|------|
   | **이름** | [`example.ipynb`](CVLface/cvlface/data_utils/recognition/eval_data/facerec_val/example.ipynb) |
   | **파일 경로** | `face_id/CVLface/cvlface/data_utils/recognition/eval_data/facerec_val/example.ipynb` |
   | **기능** | 위의 .arrow파일의 내부 구조를 확인하기 위한 테스트 코드 인덱스와 이미지, 레이블의 구성을 확인 할 수 있음 |
 
   </details>
  </details>
 <details>
   <summary><b>insightface</b></summary>
   <br>
   <details>
     <summary>fited_threshold.py</summary>
     
 | 항목 | 내용 |
 |------|------|
 | **이름** | [`fited_threshold.py`](instightface/recognition/arcface_torch/fixed_threshold.py) |
 | **파일 경로** | `face_id/instightface/recognition/arcface_torch/eval_pairs_fixed_thresh.py` |
 | **기능** | 사전 학습된 얼굴 임베딩 모델을 로드한 뒤 `gen/`·`imp/` 페어에 대해 **고정 Threshold**(315라인에 전역변수로 임의 지정) 로 LFW-스타일 검증을 수행하고 Accuracy·FP(gen의 오답 개수)·FN(imp의 오답 개수)를 출력 |
 | **사용 모델** | `backbones.get_model()` 로 불러오는 사전학습된 arcface모델 백본  |
 | **입력 형식** | `--image-path` 경로 아래 `gen/ID/…`, `imp/ID/…` 구조 (각 폴더에 `pair1_*.jpg`, `pair2_*.jpg`) |
 | **출력 형식** |  Accuracy·FP(gen의 오답 개수)·FN(imp의 오답 개수)|
 | **Arguments** | `--model-prefix` (backbone pth/pt)<br>`--image-path` (평가용 gen/imp 루트)<br>`--result-dir` (결과 저장)<br>`--batch-size` (추론 배치 크기)<br>`--network` 백본의 사이즈(ex: r50, r100) |
 | **기능 요약** | 1. 이미지 경로 파싱 → 10-fold 분할<br>2. `DataLoader` 로 배치 추론, 임베딩 추출<br>3. 코사인 유사도 거리 → 고정 Threshold 비교<br>4. Accuracy, False Positive, False Negative 계산·출력 |
 
         
   </details>
 <details>
     <summary>eval_age_benchmark.py</summary>
     
 | 항목 | 내용 |
 |------|------|
 | **이름** | [`eval_age_benchmark.py`](insightface/recognition/arcface_torch/eval_age_benchmark.py) |
 | **파일 경로** | `face_id/insightface/recognition/arcface_torch/eval_age_benchmark.py` |
 | **기능** | 사전 학습된 얼굴 임베딩 모델을 로드한 뒤 `gen/`·`imp/` 페어에 대해 LFW-스타일 검증을 수행하고 각 fold별 최적의 threshold와 오답의 거리와 경로를 출력하고 각 fold 별 Accuracy와 평균 Accuracy, gen pair의 평균거리와 표준편차, imp pair 평균거리와 표준편차를 출력 |
 | **사용 모델** | `backbones.get_model()` 로 불러오는 사전학습된 arcface모델 백본  |
 | **입력 형식** | `--image-path` 경로 아래 `gen/ID/…`, `imp/ID/…` 구조 (각 폴더에 `pair1_*.jpg`, `pair2_*.jpg`) |
 | **출력 형식** |  각 fold별 최적의 threshold<br> 오답의 코사인 유사도 거리, threshold, 둘 간의 차이<br>오답 이미지 페어의 경로<br>각 fold 별 Accuracy <br>최종 평균 Accuracy <br>gen pair의 평균거리와 표준편차 <br>imp pair 평균거리와 표준편차|
 | **Arguments** | `--model-prefix` (backbone pth/pt)<br>`--image-path` (평가용 gen/imp 루트)<br>`--result-dir` (결과 저장)<br>`--batch-size` (추론 배치 크기)<br>`--network` 백본의 사이즈(ex: r50, r100) |
 | **기능 요약** | 1. 이미지 경로 파싱 → 10-fold 분할<br>2. `DataLoader` 로 배치 추론, 임베딩 추출<br>3. 코사인 유사도 거리 → 결과 계산·출력 |
 
         
   </details>
 
   <details>
     <summary>extract_ROC.py</summary>
     
 | 항목 | 내용 |
 |------|------|
 | **이름** | [`extract_ROC.py`](insightface/recognition/arcface_torch/extract_ROC.py) |
 | **파일 경로** | `face_id/insightface/recognition/arcface_torch/extract_ROC.py` |
 | **기능** | 사전 학습된 얼굴 임베딩 모델을 로드한 뒤 `gen/`·`imp/` 페어에 대해 LFW-스타일 검증을 수행하고 각 threshold 별 confuse_matrix 값을 .npz형태로 저장 + 이를 이용하여 AUC와 EER도 출력  |
 | **사용 모델** | `backbones.get_model()` 로 불러오는 사전학습된 arcface모델 백본  |
 | **입력 형식** | `--image-path` 경로 아래 `gen/ID/…`, `imp/ID/…` 구조 (각 폴더에 `pair1_*.jpg`, `pair2_*.jpg`) |
 | **출력 형식** |  confuse_matrix 값을 .npz형태로 저장<br>AUC와 EER|
 | **Arguments** | `--model-prefix` (backbone pth/pt)<br>`--image-path` (평가용 gen/imp 루트)<br>`--result-dir` (결과 저장)<br>`--batch-size` (추론 배치 크기)<br>`--network` 백본의 사이즈(ex: r50, r100) |
 | **기능 요약** | 1. 이미지 경로 파싱 → 10-fold 분할<br>2. `DataLoader` 로 배치 추론, 임베딩 추출<br>3. 코사인 유사도 거리 → confuse_matrix 값 계산· 각 결과출력 |
 
         
   </details>
 </details>
 
 <details>
   <summary><b>TopoFR</b></summary>
   <br>
   <details>
     <summary>eval_age_benchmark.py</summary>
 
 | 항목 | 내용 |
 |------|------|
 | **이름** | [`eval_age_benchmark.py`](TopoFR/eval_age_benchmark.py) |
 | **파일 경로** | `face_id/TopoFR/eval_age_benchmark.py` |
 | **기능** | 사전 학습된 얼굴 임베딩 모델을 로드한 뒤 `gen/`·`imp/` 페어에 대해 LFW-스타일 검증을 수행하고 각 fold별 최적의 threshold와 오답의 거리와 경로를 출력하고 각 fold 별 Accuracy와 평균 Accuracy, gen pair의 평균거리와 표준편차, imp pair 평균거리와 표준편차를 출력 |
 | **사용 모델** | `backbones.get_model()` 로 불러오는 사전학습된 TopoFR모델 백본  |
 | **입력 형식** | `--image-path` 경로 아래 `gen/ID/…`, `imp/ID/…` 구조 (각 폴더에 `pair1_*.jpg`, `pair2_*.jpg`) |
 | **출력 형식** |  각 fold별 최적의 threshold<br> 오답의 코사인 유사도 거리, threshold, 둘 간의 차이<br>오답 이미지 페어의 경로<br>각 fold 별 Accuracy <br>최종 평균 Accuracy <br>gen pair의 평균거리와 표준편차 <br>imp pair 평균거리와 표준편차|
 | **Arguments** | `--model-prefix` (backbone pth/pt)<br>`--image-path` (평가용 gen/imp 루트)<br>`--result-dir` (결과 저장)<br>`--batch-size` (추론 배치 크기)<br>`--network` 백본의 사이즈(ex: r50, r100) |
 | **기능 요약** | 1. 이미지 경로 파싱 → 10-fold 분할<br>2. `DataLoader` 로 배치 추론, 임베딩 추출<br>3. 코사인 유사도 거리 → 결과 계산·출력 |
   </details>
 <details>
     <summary>extract_ROC.py</summary>
     
 | 항목 | 내용 |
 |------|------|
 | **이름** | [`extract_ROC.py`](TopoFR/extract_ROC.py) |
 | **파일 경로** | `face_id/TopoFR/extract_ROC.py` |
 | **기능** | 사전 학습된 얼굴 임베딩 모델을 로드한 뒤 `gen/`·`imp/` 페어에 대해 LFW-스타일 검증을 수행하고 각 threshold 별 confuse_matrix 값을 .npz형태로 저장 + 이를 이용하여 AUC와 EER도 출력  |
 | **사용 모델** | `backbones.get_model()` 로 불러오는 사전학습된 TopoFR모델 백본  |
 | **입력 형식** | `--image-path` 경로 아래 `gen/ID/…`, `imp/ID/…` 구조 (각 폴더에 `pair1_*.jpg`, `pair2_*.jpg`) |
 | **출력 형식** |  confuse_matrix 값을 .npz형태로 저장<br>AUC와 EER|
 | **Arguments** | `--model-prefix` (backbone pth/pt)<br>`--image-path` (평가용 gen/imp 루트)<br>`--result-dir` (결과 저장)<br>`--batch-size` (추론 배치 크기)<br>`--network` 백본의 사이즈(ex: r50, r100) |
 | **기능 요약** | 1. 이미지 경로 파싱 → 10-fold 분할<br>2. `DataLoader` 로 배치 추론, 임베딩 추출<br>3. 코사인 유사도 거리 → confuse_matrix 값 계산· 각 결과출력 |
 
         
   </details>

<details>
     <summary>embedding_compare.py</summary>
     
 | 항목 | 내용 |
 |------|------|
 | **이름** | [`embedding_compare.py`](TopoFR/embedding_compare.py) |
 | **파일 경로** | `face_id/TopoFR/embedding_compare.py` |
 | **기능** | 두 장의 이미지를 입력으로 임베딩의 코사인 유사도 거리를 측정하여 반환|
 | **사용 모델** | `backbones.get_model()` 로 불러오는 사전학습된 TopoFR모델 백본  |
 | **입력 형식** | 'img_path1', 'img_path2' 두 장의 이미지 |
 | **출력 형식** |  임베딩 거리|
 
   </details>

<details>
     <summary>run_multi.py</summary>
 
 | 항목 | 내용 |
 |------|------|
 | **이름** | [`run_multi.py`](TopoFR/run_multi.py) |
 | **파일 경로** | `face_id/TopoFR/run_multi.py` |
 | **기능** | eval_age_benchmark.py를 목표 디렉터리의 하위 디렉터리 전부에 대해 병렬로 실행하여 log를 txt형태로 저장하는 코드|
 | **사용 모델** | `backbones.get_model()` 로 불러오는 사전학습된 TopoFR모델 백본  |
 | **입력 형식** | MODEL_PREFIX = 백본모델 경로, BATCH_SIZE = 배치사이즈, NETWORK = 백본의 모델의 크기, BASE_PARENT_DIR=목표디렉터리의 경로, PARENT_DIRS= 목표디렉터리|
 | **출력 형식** |  eval_age_benchmark.py의 결과가 eval_{PARENT_DIRS}_results 디렉터리 아래 하위 폴더 이름별 txt파일로 반환|
 | **기능요약** | 여러 개의 이미지 폴더에 대해 eval_age_benchmark.py를 병렬로 실행하는 자동화 코드 41라인의 Pool()로 Multi_level을 조정|
   </details>

 <details>
     <summary>eval_lfw_tent_benchmark.py</summary>
     
 | 항목 | 내용 |
 |------|------|
 | **이름** | [`eval_lfw_tent_benchmark.py`](TopoFR/eval_lfw_tent_benchmark.py) |
 | **파일 경로** | `face_id/TopoFR/eval_lfw_tent_benchmark.py` |
 | **기능** | 배치 크기·TENT step·백본 크기(r50 / r100/r200)를 조합해 **원본 Accuracy**와 **TENT 적용 Accuracy**를 모두 계산·비교하고 로그로 저장 |
| **사용 모델** | `backbones.get_model()` 로 불러오는 **TopoFR 사전 학습 백본** |
| **입력 형식** | `--image-path` 아래<br> `gen/ID/pair1_*.jpg`, `pair2_*.jpg`<br> `imp/ID/pair1_*.jpg`, `pair2_*.jpg` |
| **출력 형식** | `log/LFW_<dataset>_benchmark_results.txt` (평가 결과·Threshold·Accuracy 기록) |
| **Arguments** | `--model-prefix` (백본 pth/pt 경로)<br>`--image-path` (평가용 gen/imp 루트)<br>`--result-dir` (로그 저장 폴더)<br>`--batch-size` (추론 배치 크기)<br>`--network` (백본 이름: `r50`, `r100`, `r200` …) |
| **기능 요약** | 1. 이미지 경로 파싱 → 10-fold 분할<br>2. `DataLoader`로 배치 추론, 임베딩 추출<br>3. 코사인 거리 기반 Accuracy·TPR/FPR 계산<br>4. 동일 설정에서 **TENT** 적용 후 동일 지표 재계산<br>5. 두 결과를 나란히 비교해 로그 파일에 저장 |

         
   </details>

<details>
     <summary>exract_accuracy.ipynb</summary>
 
 | 항목 | 내용 |
 |------|------|
 | **이름** | [`exract_accuracy.ipynb`](TopoFR/log/extract_accuracy.ipynb) |
 | **파일 경로** | `face_id/TopoFR/log/exract_accuracy.ipynb` |
 | **기능** | tent_run으로 출력된 log파일을 노션에 입력하기 좋은 형태로 변환해 주는 코드 아래 셀의 original은 위의 셀을 보고 맞게 수정해야 함|
 | **기능요약** | 실험한 내용에 맞게 bs16-ts10값은 수정, text editor에 한번 복사하고 입력|
   </details>


<details>
<summary>tent_run.py</summary>

| 항목 | 내용 |
|------|------|
| **이름** | [`tent_run.py`](TopoFR/tent_run.py) |
| **파일 경로** | `face_id/TopoFR/tent_run.py` |
| **기능** | 하위에 존재하는 **각 corruption 폴더**(contrast_1, motion_blur_2 …)를 순회하면서,<br>각 폴더를 `eval_lfw_tent_benchmark.py` 에 넘겨 **일괄 평가** 실행 |

</details>
</details>

<details>
   <summary><b>imagecorruptions</b></summary>
<br>
<details>
<summary>corrupt_images.py</summary>

| 항목 | 내용 |
|------|------|
| **이름** | [`corrupt_images.py`](imagecorruptions/corrupt_images.py) |
| **파일 경로** | `face_id/imagecorruptions/corrupt_images.py` |
| **기능** | 이미지 폴더에 **ImageNet-C 스타일 손상**을 병렬 적용<br>‣ `imagecorruptions` 라이브러리로 〈노이즈/블러/날씨/디지털〉 등 17 손상 유형 지원<br>‣ 심각도 1 – 5 선택, 최대 `-j` 코어 병렬 처리, 진행 상황 `tqdm` 표시 |
| **사용 모델** | ― |
| **입력 형식** | `$IN_DIR/**/이미지.*` (JPG/PNG 등) |
| **출력 형식** | 옵션별<br> • `subdirs` → `$OUT/…/snow/1/image.jpg`<br> • `filename` → `$OUT/…/image_snow_1.jpg`<br> • `foldername` → `$OUT/snow_1/…/image.jpg` |
| **Arguments** | **필수**<br> `in_path` (원본 이미지 루트)<br> `out_path` (출력 루트)<br> `output_type` (`subdirs` / `filename` / `foldername`)FOLDERNAME으로 지정해야 입력한 폴더구조 그대로 출력 됨<br>**선택**<br> `-j N` (동시 코어 수, default 1)<br> `-c <types>` (손상 목록 지정)<br> `-su <subset>` (`common` / `noise` / `blur` / …)<br> `-se <levels>` (심각도 리스트, default 1-5) |
| **기능 요약** | 1. 입력 폴더에서 재귀적으로 이미지 경로 수집<br>2. 선택한 손상 & 심각도별로 출력 경로 생성<br>3. 멀티프로세싱으로 `imagecorruptions.corrupt` 적용·저장<br>4. `tqdm` 진행바로 실시간 상태 표시 |

</details>

   
</details>


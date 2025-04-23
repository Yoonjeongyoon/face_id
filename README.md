# face_id  

---


<details>
  <summary><b>Dataset</b></summary>
  <br>
<details>
<br>
  
<summary><b>face_align.ipynb</b></summary>

| 항목 | 내용 |
|------|------|
| **이름** | [`face_align.ipynb`](face_id/face_align.ipynb) |
| **파일 경로** | `face_id/face_align.ipynb` |
| **기능** | 얼굴 정렬(Alignment) 수행<br>`insightface`의 `FaceAnalysis` 로 얼굴 검출 후 정렬된 이미지를 저장 |
| **사용 모델** | `insightface` 내장 face detection + landmark (CPU) |
| **입력 형식** | 단일 face 이미지가 있는 `lowdata/` 폴더<br>└─ 서브폴더 포함 전체 이미지 탐색 |
| **출력 형식** | 정렬된 얼굴 이미지 (`Asian_celebrity_align/`)<br>└─ 입력 폴더와 동일한 디렉터리 구조 |
| **기능 요약** | - 이미지 내 얼굴 검출<br>- 랜드마크 기반 얼굴 정렬<br>- 이미 처리된 파일 스킵 |
</details>
<details>
<summary><b>gen_pairs.ipynb</b></summary>

| 항목 | 내용 |
|------|------|
| **이름** | [`gen_pairs.ipynb`](face_id/gen_pairs.ipynb) |
| **파일 경로** | `face_id/gen_pairs.ipynb` |
| **기능** | 동일 인물 폴더 내 모든 이미지 조합을 생성한 뒤 무작위 3,000 쌍을 `gen/` 폴더에 복사하는 **genuine 페어 생성** 코드 |
| **사용 모델** | ― |
| **입력 형식** | `face_align.ipynb` 에서 정렬된 얼굴 이미지 (`Asian_celebrity_align/…`) |
| **출력 형식** | `gen/0/`, `gen/1/` … `gen/2999/`<br>└─ 각 폴더에 `pair1_<파일명>.jpg`, `pair2_<파일명>.jpg` |
| **기능 요약** | - 이미지 ≥2장인 인물 폴더에서 모든 조합 생성<br>- `random.seed(42)` 로 섞어 3,000 쌍 선정<br>- 쌍마다 고유 인덱스 폴더 생성 후 두 이미지를 복사 |
</details>
<details>
<summary><b> imp_pairs.py</b></summary>
| 항목 | 내용 |
|------|------|
| **이름** | [`imp_pairs.py`](face_id/imp_pairs.py) |
| **파일 경로** | `face_id/imp_pairs.py` |
| **기능** | **impostor(타인) 페어 생성** 스크립트.<br>성별(m / w)을 기준으로 **다른 인물**‑이미지 두 장을 무작위로 골라 3,000 쌍을 `imp/` 폴더에 복사 |
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
  | **파일 경로** | `CVLface/cvlface/data_utils/recognition/eval_data/facerec_val/folder_to_huggingface.py` |
  | **기능** | `gen/`·`imp/` 페어를 LFW 프로토콜에 맞게 **10‑fold** 로 분할 후, Hugging Face `Dataset`으로 변환·병합하여 adaface를 돌릴 수 있는 형태로 변환<br> 이후 evaluations/configs의 yaml파일을 수정하여 eval.py를 실행 |
  | **사용 모델** | 
  | **입력 형식** | `gen/` & `imp/` 폴더 구조 (`pair1_*.jpg`, `pair2_*.jpg`)<br>예) `gen/0/…`, `imp/42/…` |
  | **출력 형식** | `.Arrow 포맷 + 예시이미지`examples/0.jpg` … `4.jpg` |
  | **Arguments** | `--src_dataset_dir` (원본 gen/imp 경로)<br>`--output_base_dir` (HF dataset 저장 루트) |
  | **기능 요약** | 1. `split_into_folds` → gen/imp를 10개 fold로 디렉터리 복사<br>2. 각 fold를 `Dataset.from_list`로 변환하며 **전역 인덱스** 부여<br>3. `concatenate_datasets`로 병합하여 하나의 형태로 저장<br>4. 예시 이미지 5장을 `examples/`에 저장 |
  
  </details>

  <details>
    <summary>example.ipynb</summary>
    
  | 항목 | 내용 |
  |------|------|
  | **이름** | [`example.ipynb`](CVLface/cvlface/data_utils/recognition/eval_data/facerec_val/example.ipynb) |
  | **파일 경로** | `CVLface/cvlface/data_utils/recognition/eval_data/facerec_val/example.ipynb` |
  | **기능** | 위의 .arrow파일의 내부 구조를 확인하기 위한 테스트 코드 인덱스와 이미지, 레이블의 구성을 확인 할 수 있음 |

  </details>

</details>

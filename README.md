# face_id
이름 | face_align.ipynb
파일 경로 | face_id/face_align.ipynb
기능 | 얼굴 정렬 (alignment) 수행 코드. insightface의 FaceAnalysis를 사용하여 얼굴 검출 후 정렬된 이미지를 저장
사용 모델 | insightface의 내장 face detection + landmark (CPU 사용)
입력 디렉토리 | Asian_celebrity/└─ 서브폴더 포함 전체 이미지 탐색
출력 디렉토리 | Asian_celebrity_align/└─ 입력 폴더와 동일한 구조로 정렬된 얼굴 이미지 저장
기능 요약 | - 이미지 내 얼굴 검출- 랜드마크 기반 얼굴 정렬- 중복 처리 방지 (이미 처리된 파일은 스킵)




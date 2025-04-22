import os
import shutil
import random

# 경로 설정
input_folder = "Asian_celebrity_align"
output_folder = "imp"
os.makedirs(output_folder, exist_ok=True)

male_folders = []
female_folders = []
person_images = {}

# 사람별 이미지 리스트 저장 및 성별 구분
for person_folder in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, person_folder)
    if not os.path.isdir(folder_path):
        continue

    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    if not image_files:
        continue

    gender = image_files[0].split('_')[-1].split('.')[0]
    person_images[person_folder] = image_files

    if gender == 'm':
        male_folders.append(person_folder)
    elif gender == 'w':
        female_folders.append(person_folder)

print(f"Male folders: {male_folders}")
print(f"Female folders: {female_folders}")

# 쌍 만들기 전 체크
gender_groups = []
if len(male_folders) >= 2:
    gender_groups.append(male_folders)
if len(female_folders) >= 2:
    gender_groups.append(female_folders)

pair_list = []

if not gender_groups:
else:
    while len(pair_list) < 3000:
        gender_group = random.choice(gender_groups)

        if len(gender_group) < 2:
            gender_groups.remove(gender_group)
            continue

        person1, person2 = random.sample(gender_group, 2)
        img1 = random.choice(person_images[person1])
        img2 = random.choice(person_images[person2])

        pair_list.append((person1, img1, person2, img2))

print(f"생성된 pair_list: {pair_list}")

# 복사
for idx, (person1, img1, person2, img2) in enumerate(pair_list):
    pair_folder = os.path.join(output_folder, str(idx))
    os.makedirs(pair_folder, exist_ok=True)

    shutil.copy(os.path.join(input_folder, person1, img1),
                os.path.join(pair_folder, f"pair1_{img1}"))
    shutil.copy(os.path.join(input_folder, person2, img2),
                os.path.join(pair_folder, f"pair2_{img2}"))

    print(f"Pair {idx} 복사 완료: {img1}, {img2}")

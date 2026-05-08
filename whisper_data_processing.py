from google.colab import drive
drive.mount('/content/drive')

import tarfile
import os
import shutil
from tqdm import tqdm

# 설정
tar_path = "/content/drive/MyDrive/whisper_data/TS_kor_free_01.tar"
extract_dir = "/content/audio_subset"
os.makedirs(extract_dir, exist_ok=True)

# 추출 개수 설정
MAX_FILES = 100000
extracted = 0

with tarfile.open(tar_path, "r") as tar:
    print(f" {MAX_FILES}개의 .wav 파일만 추출합니다...\n")
    for member in tqdm(tar, desc="파일 검색 중"):
        # .wav 파일만 필터링
        if member.isfile() and member.name.endswith(".wav") and "__MACOSX" not in member.name:
            filename = os.path.basename(member.name)
            target_path = os.path.join(extract_dir, filename)
            with tar.extractfile(member) as src, open(target_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
            if extracted >= MAX_FILES:
                break

print(f"\n 추출 완료! 총 {extracted}개의 .wav 파일이 {extract_dir}에 저장되었습니다.")

import os
import tarfile
from tqdm import tqdm
import shutil

# 경로 설정
tar_path = "/content/drive/MyDrive/whisper_data/TL_kor_free_01.tar"
extract_dir = "/content/labels"

# 디렉토리 준비
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir, exist_ok=True)

# 압축 해제 시작
with tarfile.open(tar_path, "r") as tar:
    members = [m for m in tar.getmembers() if m.isfile()]
    print(f" 총 {len(members)}개 파일 압축 해제 중...")
    with tqdm(total=len(members), desc="압축 해제 중", unit="file") as pbar:
        for member in members:
            tar.extract(member, path=extract_dir)
            pbar.update(1)

print(f"\n 라벨 데이터 전체 해제 완료 → {extract_dir}")

import os
import json
from glob import glob

label_dir = "/content/labels"
audio_dir = "/content/audio_subset"

# 실제 오디오 파일 이름들 (집합으로 저장)
audio_files = set(os.listdir(audio_dir))

paired_data = []

json_files = glob(os.path.join(label_dir, "**/*.json"), recursive=True)

for json_path in json_files:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            file_name = data.get("File", {}).get("FileName")  # ex: K0001...wav
            transcription = data.get("Transcription", {}).get("LabelText")
            if file_name in audio_files and transcription:
                paired_data.append({
                    "audio": os.path.join(audio_dir, file_name),
                    "text": transcription.strip()
                })
    except Exception as e:
        continue  # 깨진 파일은 무시

print(f" 매칭된 오디오-텍스트 쌍 수: {len(paired_data)}개")

!pip install datasets

import os
import json
from glob import glob
from datasets import Dataset

label_dir = "/content/labels"
audio_dir = "/content/audio_subset"

audio_files = set(os.listdir(audio_dir))
paired_data = []

json_files = glob(os.path.join(label_dir, "**/*.json"), recursive=True)

for json_path in json_files:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            file_name = data.get("File", {}).get("FileName")
            transcription = data.get("Transcription", {}).get("LabelText")
            if file_name in audio_files and transcription:
                paired_data.append({
                    "audio": os.path.join(audio_dir, file_name),
                    "text": transcription.strip()
                })
    except Exception:
        continue

# Hugging Face Dataset으로 변환
dataset = Dataset.from_list(paired_data)
print(f" Whisper 학습용 데이터셋 생성 완료: {len(dataset)}개 샘플")
dataset.shuffle(seed=42).select(range(5))
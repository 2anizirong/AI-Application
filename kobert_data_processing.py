from google.colab import drive
import tarfile
import os
import json


# Google Drive 마운트
drive.mount('/content/drive')

# tar 파일 경로 (예: Google Drive에 있는 경우)
tar_path = "/content/drive/MyDrive/아동 음성 데이터셋/TL_kor_free_01.tar"
extract_path = "/content/unlockedDataset/free01"

# 압축 해제
with tarfile.open(tar_path, "r:*") as tar:
    tar.extractall(path=extract_path)
import json
import os
from tqdm import tqdm
from multiprocess import Pool, cpu_count

# 대상 폴더 내 모든 json 파일 경로 수집
json_file_paths = []
for root, _, files in os.walk(extract_path):
    for file in files:
        if file.endswith('.json'):
            json_file_paths.append(os.path.join(root, file))

# 각 json 파일에서 필요한 정보 추출
def parse_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            sentence = data['Transcription']['LabelText']
            age = int(data['Speaker']['Age'])
            age_group = data['Speaker']['AgeGroup']
            return {"sentence": sentence, "age": age, "age_group": age_group}
    except:
        return None  # 잘못된 파일 예외 처리

# 병렬 처리 수행
with Pool(cpu_count()) as pool:
    extracted_data = list(tqdm(pool.imap(parse_json, json_file_paths), total=len(json_file_paths)))

# None 제거
extracted_data = [d for d in extracted_data if d is not None]

from sklearn.utils import resample
import pandas as pd

# 원본 데이터 불러오기
df = pd.DataFrame(extracted_data)
df = df[df['age'].between(7, 12)]  # 6세 제거

# 균형 잡을 수
target_per_class = 47000
balanced_df = pd.DataFrame()

for age in range(7, 13):
    group = df[df['age'] == age]
    sampled = resample(
        group,
        replace=(len(group) < target_per_class),
        n_samples=target_per_class,
        random_state=42
    )
    balanced_df = pd.concat([balanced_df, sampled])

# 저장
balanced_df.to_csv("/content/drive/MyDrive/balanced_7to12_282k.csv", index=False)

from tqdm import tqdm
from multiprocess import Pool, cpu_count
import pandas as pd

# DataFrame으로 변환
df = pd.DataFrame(extracted_data)

print(df['age'].value_counts().sort_index())

print(balanced_df['age'].value_counts().sort_index())

!pip install --upgrade transformers
!pip install sentencepiece
!pip install gluonnlp

from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

from sklearn.model_selection import train_test_split

# 1. 균형 잡힌 CSV 불러오기
df = pd.read_csv("/content/drive/MyDrive/balanced_7to12_282k.csv")

# 2. 나이 정수 → 클래스 인덱스로 매핑 (7세: 0, ..., 12세: 5)
df['label'] = df['age'] - 7  # 7세가 클래스 0이 되도록

# 3. 학습/검증 분리
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 4. KoBERT 토크나이징
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

import torch

class KoBERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }
        return item

from transformers import Trainer, TrainingArguments, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# 1. KoBERT 모델 불러오기 (클래스 수: 6)
model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=6)

# 2. 학습 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_dir='./logs',
    logging_steps=10,
    do_eval=True,
    do_train=True,
    save_steps=500
)

# 3. 성능 측정 함수 (정확도 + F1)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

# 4. Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

%env CUDA_LAUNCH_BLOCKING=1
print("train_labels:", sorted(set(train_labels)))
print("val_labels:", sorted(set(val_labels)))

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)

model.save_pretrained("/content/kobert_age_model")
tokenizer.save_pretrained("/content/kobert_age_model")

from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/kobert_age_model /content/drive/MyDrive/kobert_age_model_backup
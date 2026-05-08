!pip install transformers datasets torchaudio librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 모델과 프로세서 로드
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

import torchaudio

def prepare_dataset(batch):
    input_features = []
    labels = []
    for audio_path, text in zip(batch["audio"], batch["text"]):
        try:
            speech_array, sr = torchaudio.load(audio_path)
            if sr != 16000:
                speech_array = torchaudio.transforms.Resample(sr, 16000)(speech_array)
            inputs = processor(speech_array[0], sampling_rate=16000, return_tensors="pt")
            input_features.append(inputs.input_features.squeeze(0))
            labels.append(processor.tokenizer(text).input_ids)
        except:
            input_features.append(None)
            labels.append(None)
    return {"input_features": input_features, "labels": labels}

valid_dataset = dataset

import math
import gc

# 나누고 싶은 단위 설정 (예: 약 8000개씩)
chunk_size = 5000
num_chunks = math.ceil(len(valid_dataset) / chunk_size)

# 전처리 함수
def process_and_save(split, idx):
    processed = split.map(
        prepare_dataset,
        batched=True,
        batch_size=8,
        remove_columns=["audio", "text"],
        num_proc=1,
        desc=f"Whisper 전처리 중 - Part {idx}"
    )
    processed.save_to_disk(f"/content/drive/MyDrive/whisper_cached_part_{idx}")
    del processed
    gc.collect()

# 자동 분할 및 전처리 실행
for idx in range(num_chunks):
    start = idx * chunk_size
    end = min(start + chunk_size, len(valid_dataset))
    split = valid_dataset.select(range(start, end))
    process_and_save(split, idx)

!pip install --no-cache-dir accelerate==0.24.1

from datasets import load_from_disk, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

dataset_paths = [f"/content/drive/MyDrive/whisper_cached_part_{i}" for i in range(10)]

def load_one(path):
    return load_from_disk(path)

loaded_datasets = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(load_one, path): path for path in dataset_paths}
    for future in tqdm(as_completed(futures), total=len(futures), desc="데이터 로드 중"):
        try:
            dataset = future.result()
            loaded_datasets.append(dataset)
        except Exception as e:
            print(f"{futures[future]} 로딩 실패: {e}")

# 병합 및 저장
train_dataset = concatenate_datasets(loaded_datasets)
train_dataset.save_to_disk("/content/drive/MyDrive/whisper_combined")

print("\n병합 완료 및 whisper_combined 폴더로 저장됨.")

from datasets import load_from_disk

train_dataset = load_from_disk("/content/drive/MyDrive/whisper_combined")

print("whisper_combined에서 train_dataset 로드 완료:", len(train_dataset), "샘플")

from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")

!pip install --upgrade --no-cache-dir transformers==4.35.2

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class CustomDataCollator:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch

# Huggingface Dataset 객체에서 상위 1만 개만 선택
mini_train_dataset = train_dataset.select(range(10000))

# 학습에 사용하지 않은 다음 1만 개 데이터셋
next_10k_dataset = train_dataset.select(range(10000, 20000))

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/whisper_finetuned_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    num_train_epochs=5,
    fp16=True,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
)

# Whisper 전용 Data Collator 설정
data_collator = CustomDataCollator(processor=processor)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=next_10k_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("/content/drive/MyDrive/whisper_finetuned_model")
processor.save_pretrained("/content/drive/MyDrive/whisper_finetuned_model")

import evaluate
from tqdm import tqdm
import torch
import re

wer_metric = evaluate.load("wer")

def trim_to_first_sentence(text):
    match = re.search(r'[.?!]', text)
    if match:
        return text[:match.end()]
    return text

unseen_dataset = train_dataset.select(range(23000, 50000))
test_subset = unseen_dataset.select(range(10))

references = []
predictions = []

for sample in tqdm(test_subset, desc="모델 추론 중"):
    input_features = torch.tensor(sample["input_features"]).unsqueeze(0).float().to(model.device)

    with torch.no_grad():
        pred_ids = model.generate(
            input_features,
            max_length=128,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            length_penalty=1.0,
            num_beams=5,
            early_stopping=True
        )

    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    pred_text = trim_to_first_sentence(pred_text)
    predictions.append(pred_text)

    label_ids = sample["labels"]
    label_ids = [id if id != -100 else processor.tokenizer.pad_token_id for id in label_ids]
    ref_text = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
    references.append(ref_text)

    print("정답:", ref_text)
    print("예측:", pred_text)
    print("———")

wer = wer_metric.compute(predictions=predictions, references=references)
print(f"\n전체 WER (Word Error Rate): {wer:.2%}")
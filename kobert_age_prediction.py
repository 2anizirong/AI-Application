def predict_age(sentence: str):
    # 모델이 올라간 디바이스 확인
    device = next(model.parameters()).device

    # 토크나이징 + device 이동
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 예측
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return predicted_class + 6  # 클래스 → 나이


test_sentence = "아"
predicted_age = predict_age(test_sentence)
print(f"예측된 나이: {predicted_age}세")


import numpy as np
unique, counts = np.unique(val_labels, return_counts=True)
print(dict(zip(unique, counts)))  # 라벨 0~6 분포 확인


preds = []
for s in val_texts[:100]:
    preds.append(predict_age(s))

import collections
print(collections.Counter(preds))
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "나이 예측 함수"
      ],
      "metadata": {
        "id": "RTHqhPq1b893"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def predict_age(sentence: str):\n",
        "    # 모델이 올라간 디바이스 확인\n",
        "    device = next(model.parameters()).device\n",
        "\n",
        "    # 토크나이징 + device 이동\n",
        "    inputs = tokenizer(\n",
        "        sentence,\n",
        "        return_tensors=\"pt\",\n",
        "        truncation=True,\n",
        "        padding=True,\n",
        "        max_length=64\n",
        "    )\n",
        "    inputs = {k: v.to(device) for k, v in inputs.items()}\n",
        "\n",
        "    # 예측\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        outputs = model(**inputs)\n",
        "        predicted_class = torch.argmax(outputs.logits, dim=1).item()\n",
        "\n",
        "    return predicted_class + 6  # 클래스 → 나이\n",
        "\n"
      ],
      "metadata": {
        "id": "og-XgO1Ib_5C"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_sentence = \"아\"\n",
        "predicted_age = predict_age(test_sentence)\n",
        "print(f\"예측된 나이: {predicted_age}세\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rWJLxxGccCEY",
        "outputId": "59d1e39a-2b12-462b-ef28-7fd927a75df6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "예측된 나이: 9세\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "검증 데이터의 정답 라벨 분포 보기"
      ],
      "metadata": {
        "id": "TsGKtC__cZSC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "unique, counts = np.unique(val_labels, return_counts=True)\n",
        "print(dict(zip(unique, counts)))  # 라벨 0~6 분포 확인\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "evk3jegYcauJ",
        "outputId": "00e00246-29c1-4813-9e6f-eafba0b36021"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{np.int64(0): np.int64(283), np.int64(1): np.int64(12336), np.int64(2): np.int64(13910), np.int64(3): np.int64(19678), np.int64(4): np.int64(13602), np.int64(5): np.int64(11191), np.int64(6): np.int64(9396)}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "preds = []\n",
        "for s in val_texts[:100]:\n",
        "    preds.append(predict_age(s))\n",
        "\n",
        "import collections\n",
        "print(collections.Counter(preds))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zseIohuacgk7",
        "outputId": "80194cca-585f-4be3-ecda-da5972832588"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Counter({9: 100})\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "예측 테스트"
      ],
      "metadata": {
        "id": "MJ7d2qRx9fRL"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "✅ 전체 요약 흐름\n",
        "📊 DataFrame에서 sentence/age 추출 (df['sentence'], df['age'])\n",
        "\n",
        "🧠 age - 6 → 클래스 라벨로 변환 (총 7개 클래스: 6~12세)\n",
        "\n",
        "🔤 KoBERT 토크나이저로 문장 인코딩\n",
        "\n",
        "🔁 Trainer로 KoBERT 분류 모델 학습 (클래스 수 = 7)\n",
        "\n",
        "🔍 새로운 문장을 입력하면 나이 예측"
      ],
      "metadata": {
        "id": "4XCkba3K9roD"
      }
    }
  ]
}
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "machine_shape": "hm",
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "JSON 데이터 병렬로 추출 전처리 작업"
      ],
      "metadata": {
        "id": "LU0whRIT-U0o"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import json\n",
        "import os\n",
        "from tqdm import tqdm\n",
        "from multiprocess import Pool, cpu_count"
      ],
      "metadata": {
        "id": "VGaaCsE3-j9U"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 대상 폴더 내 모든 json 파일 경로 수집\n",
        "json_file_paths = []\n",
        "for root, _, files in os.walk(extract_path):\n",
        "    for file in files:\n",
        "        if file.endswith('.json'):\n",
        "            json_file_paths.append(os.path.join(root, file))\n"
      ],
      "metadata": {
        "id": "d1xDp2VP-ld6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        " 각 json 파일에서 필요한 정보 추출\n",
        "def parse_json(path):\n",
        "    try:\n",
        "        with open(path, 'r', encoding='utf-8') as f:\n",
        "            data = json.load(f)\n",
        "            sentence = data['Transcription']['LabelText']\n",
        "            age = int(data['Speaker']['Age'])\n",
        "            age_group = data['Speaker']['AgeGroup']\n",
        "            return {\"sentence\": sentence, \"age\": age, \"age_group\": age_group}\n",
        "    except:\n",
        "        return None  # 잘못된 파일 예외 처리"
      ],
      "metadata": {
        "id": "Vd9w1mh6-nSt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 병렬 처리 수행\n",
        "with Pool(cpu_count()) as pool:\n",
        "    extracted_data = list(tqdm(pool.imap(parse_json, json_file_paths), total=len(json_file_paths)))"
      ],
      "metadata": {
        "id": "rwQ0taZL-qk2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# None 제거\n",
        "extracted_data = [d for d in extracted_data if d is not None]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "buV2iaFF6pAs",
        "outputId": "7a4e2929-6160-4c97-e736-78309f8fb248"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 404375/404375 [01:52<00:00, 3578.62it/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "(7~12세, 각 47,000개)"
      ],
      "metadata": {
        "id": "A5gP1j--ebKf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.utils import resample\n",
        "import pandas as pd\n",
        "\n",
        "# 원본 데이터 불러오기\n",
        "df = pd.DataFrame(extracted_data)\n",
        "df = df[df['age'].between(7, 12)]  # 6세 제거\n",
        "\n",
        "# 균형 잡을 수\n",
        "target_per_class = 47000\n",
        "balanced_df = pd.DataFrame()\n",
        "\n",
        "for age in range(7, 13):\n",
        "    group = df[df['age'] == age]\n",
        "    sampled = resample(\n",
        "        group,\n",
        "        replace=(len(group) < target_per_class),\n",
        "        n_samples=target_per_class,\n",
        "        random_state=42\n",
        "    )\n",
        "    balanced_df = pd.concat([balanced_df, sampled])\n",
        "\n",
        "# 저장\n",
        "balanced_df.to_csv(\"/content/drive/MyDrive/balanced_7to12_282k.csv\", index=False)\n"
      ],
      "metadata": {
        "id": "mKI_qfOQeYtg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "전체 데이터 연령별 분포"
      ],
      "metadata": {
        "id": "290sA8lBfIY7"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from tqdm import tqdm\n",
        "from multiprocess import Pool, cpu_count\n",
        "import pandas as pd\n",
        "\n",
        "# DataFrame으로 변환\n",
        "df = pd.DataFrame(extracted_data)\n",
        "\n",
        "print(df['age'].value_counts().sort_index())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MgSeyizs6tRv",
        "outputId": "829bc687-0fe0-48b4-d678-d4546ec59a43"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "age\n",
            "6      1500\n",
            "7     61317\n",
            "8     69037\n",
            "9     98157\n",
            "10    68434\n",
            "11    56249\n",
            "12    47282\n",
            "Name: count, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(balanced_df['age'].value_counts().sort_index())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mX9Z4SqSfDP2",
        "outputId": "c0ede2dc-bd5e-4446-c572-8ca9e37a20ef"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "age\n",
            "7     47000\n",
            "8     47000\n",
            "9     47000\n",
            "10    47000\n",
            "11    47000\n",
            "12    47000\n",
            "Name: count, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install --upgrade transformers\n",
        "!pip install sentencepiece\n",
        "!pip install gluonnlp"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qIvhNe9a9D9J",
        "outputId": "03dd7695-1cd7-44c2-bb5d-426eeed74f40"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: transformers in /usr/local/lib/python3.11/dist-packages (4.52.4)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from transformers) (3.18.0)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.30.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.32.2)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (2.0.2)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (24.2)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from transformers) (6.0.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (2024.11.6)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from transformers) (2.32.3)\n",
            "Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.21.1)\n",
            "Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.5.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.11/dist-packages (from transformers) (4.67.1)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers) (2025.3.2)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers) (4.13.2)\n",
            "Requirement already satisfied: hf-xet<2.0.0,>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers) (1.1.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.4.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2.4.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2025.4.26)\n",
            "Requirement already satisfied: sentencepiece in /usr/local/lib/python3.11/dist-packages (0.2.0)\n",
            "Requirement already satisfied: gluonnlp in /usr/local/lib/python3.11/dist-packages (0.10.0)\n",
            "Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.11/dist-packages (from gluonnlp) (2.0.2)\n",
            "Requirement already satisfied: cython in /usr/local/lib/python3.11/dist-packages (from gluonnlp) (3.0.12)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from gluonnlp) (24.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "KoBERT 모델 및 토크나이저 불러오기"
      ],
      "metadata": {
        "id": "xNHZN6BK-x32"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import BertTokenizer, BertForSequenceClassification\n",
        "import torch\n",
        "\n",
        "tokenizer = BertTokenizer.from_pretrained(\"monologg/kobert\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SCDWkDmo9H1G",
        "outputId": "5a7a4ab6-16b5-4467-aed0-6107df5b2bde"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: \n",
            "The secret `HF_TOKEN` does not exist in your Colab secrets.\n",
            "To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.\n",
            "You will be able to reuse this secret in all of your notebooks.\n",
            "Please note that authentication is recommended but still optional to access public models or datasets.\n",
            "  warnings.warn(\n",
            "The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. \n",
            "The tokenizer class you load from this checkpoint is 'KoBertTokenizer'. \n",
            "The class this function is called from is 'BertTokenizer'.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "데이터 전처리 (KoBERT 입력에 맞게)"
      ],
      "metadata": {
        "id": "R02XRZJh9Nk6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# 1. 균형 잡힌 CSV 불러오기\n",
        "df = pd.read_csv(\"/content/drive/MyDrive/balanced_7to12_282k.csv\")\n",
        "\n",
        "# 2. 나이 정수 → 클래스 인덱스로 매핑 (7세: 0, ..., 12세: 5)\n",
        "df['label'] = df['age'] - 7  # 7세가 클래스 0이 되도록\n",
        "\n",
        "# 3. 학습/검증 분리\n",
        "train_texts, val_texts, train_labels, val_labels = train_test_split(\n",
        "    df['sentence'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42\n",
        ")\n",
        "\n",
        "# 4. KoBERT 토크나이징\n",
        "train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)\n",
        "val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)\n"
      ],
      "metadata": {
        "id": "F6-dFVKL9LHR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "PyTorch Dataset 구성"
      ],
      "metadata": {
        "id": "0KOCn4239Tng"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "\n",
        "class KoBERTDataset(torch.utils.data.Dataset):\n",
        "    def __init__(self, encodings, labels):\n",
        "        self.encodings = encodings\n",
        "        self.labels = labels\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.labels)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        item = {\n",
        "            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),\n",
        "            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),\n",
        "            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)\n",
        "        }\n",
        "        return item\n",
        "\n"
      ],
      "metadata": {
        "id": "rg2GC6Bj9U_K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "KoBERT 모델 불러와 학습 설정"
      ],
      "metadata": {
        "id": "CrzIJOxB9WyB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import Trainer, TrainingArguments, BertForSequenceClassification\n",
        "from sklearn.metrics import accuracy_score, f1_score\n",
        "\n",
        "# 1. KoBERT 모델 불러오기 (클래스 수: 6)\n",
        "model = BertForSequenceClassification.from_pretrained(\"monologg/kobert\", num_labels=6)\n",
        "\n",
        "# 2. 학습 설정\n",
        "training_args = TrainingArguments(\n",
        "    output_dir='./results',\n",
        "    num_train_epochs=3,\n",
        "    per_device_train_batch_size=16,\n",
        "    per_device_eval_batch_size=64,\n",
        "    logging_dir='./logs',\n",
        "    logging_steps=10,\n",
        "    do_eval=True,\n",
        "    do_train=True,\n",
        "    save_steps=500\n",
        ")\n",
        "\n",
        "# 3. 성능 측정 함수 (정확도 + F1)\n",
        "def compute_metrics(eval_pred):\n",
        "    logits, labels = eval_pred\n",
        "    preds = logits.argmax(axis=1)\n",
        "    return {\n",
        "        \"accuracy\": accuracy_score(labels, preds),\n",
        "        \"f1\": f1_score(labels, preds, average=\"macro\")\n",
        "    }\n",
        "\n",
        "# 4. Trainer 정의\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    train_dataset=train_dataset,\n",
        "    eval_dataset=val_dataset,\n",
        "    compute_metrics=compute_metrics\n",
        ")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 514
        },
        "id": "OInE93pW9YTZ",
        "outputId": "7d072cdd-a648-46f1-f41b-79f1e8501761"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Some weights of BertForSequenceClassification were not initialized from the model checkpoint at monologg/kobert and are newly initialized: ['classifier.bias', 'classifier.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "RuntimeError",
          "evalue": "CUDA error: device-side assert triggered\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-40-24c1d56dea51>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     28\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     29\u001b[0m \u001b[0;31m# 4. Trainer 정의\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 30\u001b[0;31m trainer = Trainer(\n\u001b[0m\u001b[1;32m     31\u001b[0m     \u001b[0mmodel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     32\u001b[0m     \u001b[0margs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtraining_args\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/transformers/utils/deprecation.py\u001b[0m in \u001b[0;36mwrapped_func\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    170\u001b[0m                 \u001b[0mwarnings\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwarn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmessage\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mFutureWarning\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mstacklevel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    171\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 172\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    173\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    174\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mwrapped_func\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/transformers/trainer.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)\u001b[0m\n\u001b[1;32m    463\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcompute_loss_func\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcompute_loss_func\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    464\u001b[0m         \u001b[0;31m# Seed must be set before instantiating the model when using model\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 465\u001b[0;31m         \u001b[0menable_full_determinism\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfull_determinism\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0mset_seed\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    466\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    467\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhp_name\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/transformers/trainer_utils.py\u001b[0m in \u001b[0;36mset_seed\u001b[0;34m(seed, deterministic)\u001b[0m\n\u001b[1;32m    104\u001b[0m     \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrandom\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    105\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mis_torch_available\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 106\u001b[0;31m         \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmanual_seed\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    107\u001b[0m         \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcuda\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmanual_seed_all\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    108\u001b[0m         \u001b[0;31m# ^^ safe to call this function even if cuda is not available\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/_compile.py\u001b[0m in \u001b[0;36minner\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     30\u001b[0m                 \u001b[0mfn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__dynamo_disable\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdisable_fn\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     31\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 32\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mdisable_fn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     33\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     34\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0minner\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py\u001b[0m in \u001b[0;36m_fn\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    743\u001b[0m             )\n\u001b[1;32m    744\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 745\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    746\u001b[0m             \u001b[0;32mfinally\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    747\u001b[0m                 \u001b[0m_maybe_set_eval_frame\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mprior\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/random.py\u001b[0m in \u001b[0;36mmanual_seed\u001b[0;34m(seed)\u001b[0m\n\u001b[1;32m     44\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     45\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcuda\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_is_in_bad_fork\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 46\u001b[0;31m         \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcuda\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmanual_seed_all\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     47\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     48\u001b[0m     \u001b[0;32mimport\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmps\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/cuda/random.py\u001b[0m in \u001b[0;36mmanual_seed_all\u001b[0;34m(seed)\u001b[0m\n\u001b[1;32m    125\u001b[0m             \u001b[0mdefault_generator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmanual_seed\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    126\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 127\u001b[0;31m     \u001b[0m_lazy_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcb\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mseed_all\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    128\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    129\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/cuda/__init__.py\u001b[0m in \u001b[0;36m_lazy_call\u001b[0;34m(callable, **kwargs)\u001b[0m\n\u001b[1;32m    247\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0m_lazy_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcallable\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    248\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mis_initialized\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 249\u001b[0;31m         \u001b[0mcallable\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    250\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    251\u001b[0m         \u001b[0;31m# TODO(torch_deploy): this accesses linecache, which attempts to read the\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/cuda/random.py\u001b[0m in \u001b[0;36mcb\u001b[0;34m()\u001b[0m\n\u001b[1;32m    123\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdevice_count\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    124\u001b[0m             \u001b[0mdefault_generator\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcuda\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdefault_generators\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 125\u001b[0;31m             \u001b[0mdefault_generator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmanual_seed\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    126\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    127\u001b[0m     \u001b[0m_lazy_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcb\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mseed_all\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mRuntimeError\u001b[0m: CUDA error: device-side assert triggered\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "모델 학습"
      ],
      "metadata": {
        "id": "8tllYo629cq9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%env CUDA_LAUNCH_BLOCKING=1\n",
        "print(\"train_labels:\", sorted(set(train_labels)))\n",
        "print(\"val_labels:\", sorted(set(val_labels)))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4vZH_Uqbhbt-",
        "outputId": "c6518fcb-5b02-4975-f47b-ce020c42403c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "env: CUDA_LAUNCH_BLOCKING=1\n",
            "train_labels: [0, 1, 2, 3, 4, 5]\n",
            "val_labels: [0, 1, 2, 3, 4, 5]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "trainer.train()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 381
        },
        "id": "MlCRMHqb9don",
        "outputId": "9874a0f6-92e4-489a-ab0d-8269009a11e8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "RuntimeError",
          "evalue": "CUDA error: device-side assert triggered\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-37-3435b262f1ae>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mtrainer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/transformers/trainer.py\u001b[0m in \u001b[0;36mtrain\u001b[0;34m(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)\u001b[0m\n\u001b[1;32m   2238\u001b[0m                 \u001b[0mhf_hub_utils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menable_progress_bars\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2239\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2240\u001b[0;31m             return inner_training_loop(\n\u001b[0m\u001b[1;32m   2241\u001b[0m                 \u001b[0margs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2242\u001b[0m                 \u001b[0mresume_from_checkpoint\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mresume_from_checkpoint\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/transformers/trainer.py\u001b[0m in \u001b[0;36m_inner_training_loop\u001b[0;34m(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)\u001b[0m\n\u001b[1;32m   2248\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresume_from_checkpoint\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrial\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mignore_keys_for_eval\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2249\u001b[0m     ):\n\u001b[0;32m-> 2250\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maccelerator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfree_memory\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2251\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_train_batch_size\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2252\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mauto_find_batch_size\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/accelerate/accelerator.py\u001b[0m in \u001b[0;36mfree_memory\u001b[0;34m(self, *objects)\u001b[0m\n\u001b[1;32m   3520\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdeepspeed_engine_wrapped\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdestroy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3521\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdeepspeed_engine_wrapped\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3522\u001b[0;31m         \u001b[0mobjects\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrelease_memory\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mobjects\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3523\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_schedulers\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3524\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_optimizers\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/accelerate/utils/memory.py\u001b[0m in \u001b[0;36mrelease_memory\u001b[0;34m(*objects)\u001b[0m\n\u001b[1;32m     94\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mobjects\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     95\u001b[0m         \u001b[0mobjects\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 96\u001b[0;31m     \u001b[0mclear_device_cache\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mgarbage_collection\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     97\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mobjects\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     98\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/accelerate/utils/memory.py\u001b[0m in \u001b[0;36mclear_device_cache\u001b[0;34m(garbage_collection)\u001b[0m\n\u001b[1;32m     62\u001b[0m         \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmps\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mempty_cache\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     63\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mis_cuda_available\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 64\u001b[0;31m         \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcuda\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mempty_cache\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     65\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mis_hpu_available\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     66\u001b[0m         \u001b[0;31m# torch.hpu.empty_cache() # not available on hpu as it reserves all device memory for the current process\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/cuda/memory.py\u001b[0m in \u001b[0;36mempty_cache\u001b[0;34m()\u001b[0m\n\u001b[1;32m    216\u001b[0m     \"\"\"\n\u001b[1;32m    217\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mis_initialized\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 218\u001b[0;31m         \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_C\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_cuda_emptyCache\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    219\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    220\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mRuntimeError\u001b[0m: CUDA error: device-side assert triggered\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "eval_results = trainer.evaluate()\n",
        "print(eval_results)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 77
        },
        "id": "nEd-qKBGX7Vo",
        "outputId": "04b76857-2db1-4159-e0a1-72c78e0f741b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "    <div>\n",
              "      \n",
              "      <progress value='3771' max='1257' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
              "      [1257/1257 13:08]\n",
              "    </div>\n",
              "    "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'eval_loss': 1.781723976135254, 'eval_runtime': 41.7179, 'eval_samples_per_second': 1927.133, 'eval_steps_per_second': 30.131, 'epoch': 3.0}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.save_pretrained(\"/content/kobert_age_model\")\n",
        "tokenizer.save_pretrained(\"/content/kobert_age_model\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MTTFQyYbYYUf",
        "outputId": "3841e89d-7096-4eee-ea8c-b8af2a785bc7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "('/content/kobert_age_model/tokenizer_config.json',\n",
              " '/content/kobert_age_model/special_tokens_map.json',\n",
              " '/content/kobert_age_model/vocab.txt',\n",
              " '/content/kobert_age_model/added_tokens.json')"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "!cp -r /content/kobert_age_model /content/drive/MyDrive/kobert_age_model_backup\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "svh2_ulBYbUH",
        "outputId": "2667d7c7-fec4-40f9-f06d-97585b0b3a1b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    }
  ]
}
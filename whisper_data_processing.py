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
        "구글 드라이브 마운트"
      ],
      "metadata": {
        "id": "Bbs3nXE9yAaN"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7C2rSHBDhDvk",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5fa804a8-5d65-49cb-9304-6a0ca00bf3fa"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "TS 파일 압축 풀기"
      ],
      "metadata": {
        "id": "iE-IkGm9yHpV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import tarfile\n",
        "import os\n",
        "import shutil\n",
        "from tqdm import tqdm\n",
        "\n",
        "# 설정\n",
        "tar_path = \"/content/drive/MyDrive/whisper_data/TS_kor_free_01.tar\"\n",
        "extract_dir = \"/content/audio_subset\"\n",
        "os.makedirs(extract_dir, exist_ok=True)\n",
        "\n",
        "# 추출 개수 설정\n",
        "MAX_FILES = 100000\n",
        "extracted = 0\n",
        "\n",
        "with tarfile.open(tar_path, \"r\") as tar:\n",
        "    print(f\" {MAX_FILES}개의 .wav 파일만 추출합니다...\\n\")\n",
        "    for member in tqdm(tar, desc=\"파일 검색 중\"):\n",
        "        # .wav 파일만 필터링\n",
        "        if member.isfile() and member.name.endswith(\".wav\") and \"__MACOSX\" not in member.name:\n",
        "            filename = os.path.basename(member.name)\n",
        "            target_path = os.path.join(extract_dir, filename)\n",
        "            with tar.extractfile(member) as src, open(target_path, 'wb') as dst:\n",
        "                shutil.copyfileobj(src, dst)\n",
        "            extracted += 1\n",
        "            if extracted >= MAX_FILES:\n",
        "                break\n",
        "\n",
        "print(f\"\\n 추출 완료! 총 {extracted}개의 .wav 파일이 {extract_dir}에 저장되었습니다.\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jS4I6K5Qqizi",
        "outputId": "5e3c4b10-a299-4bd1-d95a-8041c485739c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " 100000개의 .wav 파일만 추출합니다...\n",
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "파일 검색 중: 100181it [08:18, 201.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            " 추출 완료! 총 100000개의 .wav 파일이 /content/audio_subset에 저장되었습니다.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "TL 파일 압축 풀기"
      ],
      "metadata": {
        "id": "d5ODcrp0yPF4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import tarfile\n",
        "from tqdm import tqdm\n",
        "import shutil\n",
        "\n",
        "# 경로 설정\n",
        "tar_path = \"/content/drive/MyDrive/whisper_data/TL_kor_free_01.tar\"\n",
        "extract_dir = \"/content/labels\"\n",
        "\n",
        "# 디렉토리 준비\n",
        "if os.path.exists(extract_dir):\n",
        "    shutil.rmtree(extract_dir)\n",
        "os.makedirs(extract_dir, exist_ok=True)\n",
        "\n",
        "# 압축 해제 시작\n",
        "with tarfile.open(tar_path, \"r\") as tar:\n",
        "    members = [m for m in tar.getmembers() if m.isfile()]\n",
        "    print(f\" 총 {len(members)}개 파일 압축 해제 중...\")\n",
        "    with tqdm(total=len(members), desc=\"압축 해제 중\", unit=\"file\") as pbar:\n",
        "        for member in members:\n",
        "            tar.extract(member, path=extract_dir)\n",
        "            pbar.update(1)\n",
        "\n",
        "print(f\"\\n 라벨 데이터 전체 해제 완료 → {extract_dir}\")"
      ],
      "metadata": {
        "id": "HEzvYCHa4BaH",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2a5a1e8e-1d29-428c-df89-59c71620eb99"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " 총 404375개 파일 압축 해제 중...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "압축 해제 중: 100%|██████████| 404375/404375 [00:50<00:00, 8082.78file/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            " 라벨 데이터 전체 해제 완료 → /content/labels\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "오디오-텍스트 쌍 매칭하기"
      ],
      "metadata": {
        "id": "cAjXG-ScyUum"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import json\n",
        "from glob import glob\n",
        "\n",
        "label_dir = \"/content/labels\"\n",
        "audio_dir = \"/content/audio_subset\"\n",
        "\n",
        "# 실제 오디오 파일 이름들 (집합으로 저장)\n",
        "audio_files = set(os.listdir(audio_dir))\n",
        "\n",
        "paired_data = []\n",
        "\n",
        "json_files = glob(os.path.join(label_dir, \"**/*.json\"), recursive=True)\n",
        "\n",
        "for json_path in json_files:\n",
        "    try:\n",
        "        with open(json_path, 'r', encoding='utf-8') as f:\n",
        "            data = json.load(f)\n",
        "            file_name = data.get(\"File\", {}).get(\"FileName\")  # ex: K0001...wav\n",
        "            transcription = data.get(\"Transcription\", {}).get(\"LabelText\")\n",
        "            if file_name in audio_files and transcription:\n",
        "                paired_data.append({\n",
        "                    \"audio\": os.path.join(audio_dir, file_name),\n",
        "                    \"text\": transcription.strip()\n",
        "                })\n",
        "    except Exception as e:\n",
        "        continue  # 깨진 파일은 무시\n",
        "\n",
        "print(f\" 매칭된 오디오-텍스트 쌍 수: {len(paired_data)}개\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u61HRrVSktCl",
        "outputId": "b4242000-e536-45b8-e65d-3ff03e9d365a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " 매칭된 오디오-텍스트 쌍 수: 100000개\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install datasets"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E8LKFT99x88Y",
        "outputId": "ee90b144-e3d7-417c-b16e-eae3ebe6e6de"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: datasets in /usr/local/lib/python3.11/dist-packages (2.14.4)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.11/dist-packages (from datasets) (2.0.2)\n",
            "Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.11/dist-packages (from datasets) (18.1.0)\n",
            "Requirement already satisfied: dill<0.3.8,>=0.3.0 in /usr/local/lib/python3.11/dist-packages (from datasets) (0.3.7)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (from datasets) (2.2.2)\n",
            "Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.11/dist-packages (from datasets) (2.32.3)\n",
            "Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.11/dist-packages (from datasets) (4.67.1)\n",
            "Requirement already satisfied: xxhash in /usr/local/lib/python3.11/dist-packages (from datasets) (3.5.0)\n",
            "Requirement already satisfied: multiprocess in /usr/local/lib/python3.11/dist-packages (from datasets) (0.70.15)\n",
            "Requirement already satisfied: fsspec>=2021.11.1 in /usr/local/lib/python3.11/dist-packages (from fsspec[http]>=2021.11.1->datasets) (2025.3.2)\n",
            "Requirement already satisfied: aiohttp in /usr/local/lib/python3.11/dist-packages (from datasets) (3.11.15)\n",
            "Requirement already satisfied: huggingface-hub<1.0.0,>=0.14.0 in /usr/local/lib/python3.11/dist-packages (from datasets) (0.31.4)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from datasets) (24.2)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from datasets) (6.0.2)\n",
            "Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (2.6.1)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (1.3.2)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (25.3.0)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (1.6.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (6.4.4)\n",
            "Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (0.3.1)\n",
            "Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (1.20.0)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (3.18.0)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (4.13.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.19.0->datasets) (3.4.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.19.0->datasets) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.19.0->datasets) (2.4.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests>=2.19.0->datasets) (2025.4.26)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas->datasets) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas->datasets) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas->datasets) (2025.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.17.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "whisper 학습용 데이터셋 생성 (5개 샘플)"
      ],
      "metadata": {
        "id": "MXyEyZQqyaO1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import json\n",
        "from glob import glob\n",
        "from datasets import Dataset\n",
        "\n",
        "label_dir = \"/content/labels\"\n",
        "audio_dir = \"/content/audio_subset\"\n",
        "\n",
        "audio_files = set(os.listdir(audio_dir))\n",
        "paired_data = []\n",
        "\n",
        "json_files = glob(os.path.join(label_dir, \"**/*.json\"), recursive=True)\n",
        "\n",
        "for json_path in json_files:\n",
        "    try:\n",
        "        with open(json_path, 'r', encoding='utf-8') as f:\n",
        "            data = json.load(f)\n",
        "            file_name = data.get(\"File\", {}).get(\"FileName\")\n",
        "            transcription = data.get(\"Transcription\", {}).get(\"LabelText\")\n",
        "            if file_name in audio_files and transcription:\n",
        "                paired_data.append({\n",
        "                    \"audio\": os.path.join(audio_dir, file_name),\n",
        "                    \"text\": transcription.strip()\n",
        "                })\n",
        "    except Exception:\n",
        "        continue\n",
        "\n",
        "# Hugging Face Dataset으로 변환\n",
        "dataset = Dataset.from_list(paired_data)\n",
        "print(f\" Whisper 학습용 데이터셋 생성 완료: {len(dataset)}개 샘플\")\n",
        "dataset.shuffle(seed=42).select(range(5))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yMkOr99JyEN0",
        "outputId": "0ed20033-932c-4529-9784-1f8c5ac26b1a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " Whisper 학습용 데이터셋 생성 완료: 100000개 샘플\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Dataset({\n",
              "    features: ['audio', 'text'],\n",
              "    num_rows: 5\n",
              "})"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    }
  ]
}
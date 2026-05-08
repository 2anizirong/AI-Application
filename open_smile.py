from google.colab import drive
drive.mount('/content/drive')

# STEP 2: OpenSMILE 설치 및 빌드
!apt-get install -y build-essential cmake libfftw3-dev libasound2-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libvorbis-dev libboost-all-dev

!git clone https://github.com/audeering/opensmile.git /content/opensmile
%cd /content/opensmile

!mkdir build
%cd build
!cmake ..
!make -j4  # 병렬 빌드 (속도 빠름)

import os
import glob

opensmile_bin = '/content/opensmile/build/progsrc/smilextract/SMILExtract'
config_file = '/content/opensmile/config/egemaps/v01a/eGeMAPSv01a.conf'

input_dir = '/content/drive/MyDrive/인지응연습/7938'
output_dir = '/content/drive/MyDrive/인지응연습/7938output'
os.makedirs(output_dir, exist_ok=True)

wav_files = glob.glob(os.path.join(input_dir, '*.wav'))

for wav_path in wav_files:
    filename = os.path.splitext(os.path.basename(wav_path))[0]
    output_csv = os.path.join(output_dir, f'{filename}.csv')

    print(f"▶ Processing: {filename}")
    !$opensmile_bin -C $config_file -I "$wav_path" -csvoutput "$output_csv"

csv_path = '/content/drive/MyDrive/인지응연습/7938output/K00017938-BFG23-L1N2D4-E-K0KK-04563326.csv'

import pandas as pd
df = pd.read_csv(csv_path, sep=';')

def calculate_autism_risk_score(row):
    score = 0

    # --- 음의 높낮이 변동: 기본 주파수 범위 ---
    if row['F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2'] > 3.0:
        score += 1  # 세미톤 기준으로 주파수 변동이 큼 (PFR↑)

    # --- 음의 높이 변화량: 주파수 변동성 ---
    if row['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'] > 0.3:
        score += 1  # 기본 주파수의 변동 표준편차 (vFo↑)

    # --- 진폭 변화량: 음성의 떨림 (shimmer) ---
    if row['shimmerLocaldB_sma3nz_amean'] > 1.5:
        score += 1  # 진폭의 불안정성 (vAm↑)

    # --- 평균 음의 세기: loudness 평균 ---
    if row['loudness_sma3_amean'] > 1.0:
        score += 1  # 강도가 너무 크면 ASD 특성 가능

    # --- 강도 범위: 말의 세기 변화 정도 ---
    loudness_range = row['loudness_sma3_percentile80.0'] - row['loudness_sma3_percentile20.0']
    if loudness_range > 2.5:
        score += 1  # 강도 변화가 클 경우 (dB range↑)

    # --- 최소 음의 세기: 너무 낮은 강도인지 확인 ---
    if row['loudness_sma3_percentile20.0'] < -1.0:
        score += 1  # 아주 작은 소리(속삭임 수준)는 비정상일 수 있음

    # --- 최대 음의 세기: 너무 큰 소리인지 확인 ---
    if row['loudness_sma3_percentile80.0'] > 2.0:
        score += 1  # 너무 크면 음성 조절 미숙 가능성

    # --- 평균 음압레벨 (정규화된 상대값) ---
    if row['equivalentSoundLevel_dBp'] > -15:
        score += 1  # 일반적 상대 기준 -20~-10 사이. 너무 크면 비정상

    # --- 유성(segment된 음성)의 평균 길이: 너무 짧으면 끊김 많음 ---
    if row['MeanVoicedSegmentLengthSec'] < 0.2:
        score += 1  # 한 문장의 음성 지속력이 낮음

    # --- 무성 구간 평균 길이: 무성 시간이 길면 말이 끊김처럼 들림 ---
    if row['MeanUnvoicedSegmentLength'] > 0.3:
        score += 1  # 발화 중간에 침묵이 길면 비정상 발화 가능성

    # --- 초당 음성 segment 수: 너무 많으면 끊어말함, 너무 적어도 이상 ---
    if row['VoicedSegmentsPerSec'] > 10 or row['VoicedSegmentsPerSec'] < 2:
        score += 1  # 발화 리듬의 비정상 가능성

    return score

df['autism_risk_score'] = df.apply(calculate_autism_risk_score, axis=1)
print(df[['name', 'autism_risk_score']])
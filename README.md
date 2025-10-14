# 🏋️‍♂️ Squat Posture Classification

센서 기반 스쿼트 자세 분류 프로젝트의 정리된 베이스라인 저장소입니다.  
라벨링된 윈도우 데이터를 중심으로 전처리 → 학습 → 배포까지의 파이프라인을 한 repo 안에서 관리할 수 있도록 디렉토리 구조를 재정비했습니다.

## 프로젝트 개요

- **목표**: 18채널 IMU 데이터로 5가지 스쿼트 자세를 실시간 분류
- **활용**: 기본 모델 학습, 증강/전처리 실험, 임베디드 디바이스 배포 대비
- **구성요소**:
  - `src/data_loading.py`: 데이터 로딩 및 분할 유틸리티
  - `src/augmentations.py`: 센서 시계열 증강 함수 모음
  - `src/models/`: 기본 모델 아키텍처 (CNN + GRU)
  - `src/train.py`: 명령형 학습 스크립트
  - `data/`: 원천/중간/라벨 데이터와 KU-HAR 사전학습 데이터

## 자세 클래스

| Class ID | 자세 이름 | 수집 및 라벨링 가이드 |
|----------|-----------|-----------------------|
| **0** | 정자세 (Correct) | 엉덩이가 무릎보다 낮고 척추 중립 유지 |
| **1** | 무릎 모임 (Knee Valgus) | 일어설 때 무릎이 안쪽으로 모이는 패턴 |
| **2** | 벗 윙크 (Butt Wink) | 최하단에서 골반이 안으로 말려 들어감 |
| **3** | 상체 과다 숙임 (Excessive Lean) | 상체가 바닥을 향해 과하게 숙여짐 |
| **4** | 얕은 스쿼트 (Partial Squat) | 엉덩이가 무릎 높이까지만 내려감 |

이 클래스 정의와 라벨링 기준은 `src/data_loading.py`의 `SQUAT_CLASS_GUIDE`에도 포함되어 있어 코드와 문서를 일관되게 유지합니다.

## 디렉토리 구조

```
squat_project/
├── data/
│   ├── manually_labeled/      # 라벨 확정된 윈도우 CSV
│   ├── pretrain/
│   │   └── kuhar/             # KU-HAR 원본 데이터 (외부 배포 금지)
│   ├── raw/                   # 센서 로그 등 미가공 데이터
│   ├── interim/               # 전처리 중간 산출물
│   └── processed/             # 학습용으로 가공된 데이터
├── models/                    # 학습된 가중치 및 체크포인트 (Git 무시)
├── notebooks/                 # 분석/실험용 Jupyter 노트북
├── scripts/                   # 배치 실행 스크립트
├── src/
│   ├── augmentations.py       # 증강 파이프라인 구성
│   ├── data_loading.py        # 데이터셋/데이터로더 헬퍼
│   ├── models/                # 모델 아키텍처 모듈
│   └── train.py               # 학습 엔트리 포인트
├── requirements.txt           # 의존성 목록
├── project_plan.md            # 프로젝트 일정 및 TODO (별도 관리)
└── README.md
```

빈 디렉토리는 `.gitkeep`으로 유지되고 대용량 데이터(`data/raw`, `data/pretrain`, 등)는 기본적으로 `.gitignore` 처리되어 로컬 전용으로 관리됩니다.

## 시작하기

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

데이터를 `data/` 아래에 채워 넣은 뒤 다음 명령으로 기본 학습 루프를 실행할 수 있습니다.

```bash
python -m src.train --data-root data --epochs 30 --batch-size 64
```

`src/train.py`는 간단한 AdamW + CrossEntropyLoss 파이프라인을 제공하며, GPU가 감지되면 자동으로 활용합니다. 필요에 맞게 모델 아키텍처(`src/models/temporal.py`)나 데이터 증강(`src/augmentations.py`)을 수정하세요.

## 다음 단계 제안

1. `src/models/legacy_baseline.py`를 참고해 고급 아키텍처를 모듈화
2. KU-HAR 전처리 코드를 `data/pretrain/` 파이프라인으로 정리
3. 노트북 디렉토리에서 EDA/모델 비교 실험 기록화

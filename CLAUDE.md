.venv, uv 사용

## 프로젝트 개요
- 6G O-RAN 환경에서 task-agnostic site representation을 통한 채널 추정 연구
- Sionna RT 기반 mmWave 채널 데이터셋 (뮌헨 도시 환경)
- EIS Lab (연세대 고정길 교수 연구실) 소속

## 디렉토리 구조
```
src/
├── config.py                    # 시뮬레이션/모델 설정
├── data/                        # 데이터 생성(generate.py), 로더(dataset.py), 유틸(utils.py), 씬분석(analyze_munich_scene.py)
├── models/
│   ├── estimator.py             # 3-way 모델 (E + theta_task + theta_BS)
│   ├── adapters.py              # Site adapter (SSF, LoRA 등)
│   └── baselines.py             # 비교 모델
├── training/
│   ├── trainer.py               # 학습 루프
│   ├── federated.py             # FL 학습
│   └── meta_learning.py         # Meta-learning (MAML 등)
├── experiments/                 # 실험별 폴더 (train.py + results.ipynb)
│   ├── E0_baseline/            # Baseline sanity check + architecture search
│   ├── E1_fl_verification/     # FL comparison (3-way vs FedPer vs FedAvg)
│   ├── E2_task_agnostic/       # Task-agnostic transfer
│   └── E3_ablation/            # Ablation studies
├── dataset_operation/           # 데이터 생성/로드/비교/분석
└── tracker.py                   # 실험 추적 (Tracker 클래스)

assets/
├── configs/                     # Scene preset YAML (munich_uma8, munich_umi16 등)
├── data/                        # 채널 데이터 (channels/, channels_umi16/)
├── checkpoints/                 # 학습된 모델 체크포인트 (phase별 하위폴더)
├── plots/                       # 실험별 시각화 (0_baseline/, 1_fl_verification/ 등)
├── results/                     # 평가 결과 JSON (comparison/, phase0/ 등)
├── runs/                        # 실험 트래커 데이터 (run.json, metrics.jsonl, output.log)
└── backlog.json                 # 실험 큐 (대시보드 + /backlog 공유 DB)

docs/
├── research_and_experiments.md  # 연구 아이디어, 실험 설계, findings 종합
├── relworks.md                  # Related works 분석 (46+ papers)
└── papers/                      # 참고 논문 PDF (92개) + txt/ (논문 텍스트 추출)
web/
├── dashboard.py                # 대시보드 서버
└── static/                     # HTML, CSS, JS
```

## 실험 추적 (Tracker)

실험 스크립트 작성 시 반드시 `Tracker`를 사용하여 자동 추적.
Tracker는 하나 (`src/tracker.py`) — 사용 패턴이 두 가지:

```python
from src.tracker import Tracker
from src.training.trainer import train_local

# 패턴 1: train_local() 위임 (E1+, 단일 모델 학습)
# train_local() 내부에서 epoch 단위로 tracker.log() 자동 호출
with Tracker("E1/uma/fedavg", config={...}, capture_output=True) as run:
    result = train_local(model, train_loader, val_loader,
                         tracker=run, epochs=100, lr=1e-3)

# 패턴 2: 수동 로깅 (E0 FL, 커스텀 loop)
# FL round 등 직접 제어하는 loop에서 run.log() 직접 호출
with Tracker("E0/uma/A_ssf_E", config={...}, capture_output=True) as run:
    for rnd in range(fl_rounds):
        for bs in all_bs:
            train_epoch(models[bs], loaders[bs], optimizers[bs], device)
        aggregate(models, shared_keys)
        avg_db = evaluate_all(models, val_loaders)
        run.log(round=rnd, avg_val_nmse_db=avg_db)  # 직접 호출
    run.set_result(avg_nmse_db=final_avg)
```

**규칙:**
- `name` 형식: `"E{N}/{dataset}/{config}"` (예: `E0/uma/A_ssf_E`, `E1/umi/fedavg`)
- `config`에 주요 하이퍼파라미터 전달 (대시보드에서 비교용)
- `capture_output=True` 항상 사용 (stdout/stderr → `output.log`)
- `train_local()` 호출 시 `tracker=run` 인자 필수
- FL loop에서는 round 단위로 `run.log()` 직접 호출
- 대시보드 실행: `python web/dashboard.py` → http://localhost:8765

**실험 설계 메타데이터 (experiment-level):**

Tracker 생성 시 `purpose`, `variables`, `hypothesis`, `eval_criteria`를 전달하면
`assets/runs/.experiments/{experiment}.json`에 실험 단위로 자동 저장.
대시보드 Experiments 탭의 overview에 표시됨 (run detail이 아닌 experiment 단위).

```python
with Tracker("E0/uma/A_ssf_E", config={...}, capture_output=True,
    purpose="Architecture search: structure × adapter × placement",
    variables={
        "independent": ["adapter", "placement", "sharing"],
        "dependent": ["avg_nmse_db"],
        "controlled": ["fl_rounds=50", "lr=1e-3"],
    },
    hypothesis="Which adapter × placement × sharing combo gives lowest test NMSE?",
    eval_criteria="Compare avg_test_bs NMSE across configs; best = lowest dB",
) as run:
```

Hypothesis 유형:
- **Confirmatory**: `"LoRA+encoder_only sharing이 best test NMSE 달성"`
- **Exploratory**: `"Which adapter × placement combo gives lowest test NMSE?"`
- **Threshold**: `"최소 하나의 config가 test NMSE < -17 dB 달성"`

`git_commit`은 자동 기록됨. 같은 experiment의 여러 run이 동일 메타를 공유.

## Backlog (통합 작업 관리)

`assets/backlog.json`은 모든 작업의 single source of truth.
실험, 구현, 조사, 버그 수정 등 모든 종류의 작업을 통합 관리.
대시보드 Queue 탭과 `/backlog` 커맨드가 동일 파일을 읽고 씀.

```bash
# Claude Code에서
/backlog list              # 전체 작업 확인
/backlog add phase1/test   # 작업 추가
/backlog done abc123 "완료 요약"  # 작업 완료 처리
/backlog run next          # 다음 실험 실행
```

Backlog item 스키마:
```json
{
  "id": "8자리 hex",
  "name": "implement/sidebar-toggle",
  "type": "implement",
  "description": "설명",
  "script": null,
  "config": {},
  "priority": "normal",
  "status": "queued",
  "summary": null,
  "session": "a1b2c3"
}
```

`session` 필드는 Claude 세션별 6자리 ID (`/tmp/claude_task_session`에서 관리).
대시보드 오른쪽 사이드바에서 세션별 태스크 구분 가능.

## 작업 추적 (Task Logging)

모든 **의미 있는 작업**은 `assets/backlog.json`에 기록한다.
대시보드 Queue 탭에서 실험과 함께 통합 관리됨.

### 언제 기록하는가
- 코드 구현/수정 (새 기능, 리팩토링, 버그 수정)
- 조사/분석 (논문 리뷰, 코드 분석, 성능 분석)
- 설정/인프라 변경 (대시보드, 환경설정, 디렉토리 구조)
- 실험 실행 (기존 backlog flow)

**기록하지 않는 것:** 단순 질문 응답, 파일 읽기만, 한두 줄 수정

### 어떻게 기록하는가
작업 시작 시 backlog에 추가, 완료 시 상태 업데이트:

```json
{
  "id": "8자리 hex",
  "name": "implement/sidebar-toggle",
  "type": "implement",
  "description": "사이드바 접기/펴기 토글 구현",
  "status": "in_progress",
  "summary": null,
  "created_at": "ISO timestamp"
}
```

### 작업 흐름 (Quick Task Logging)
가장 빠른 방법은 `/task` 커맨드:
```bash
/task start implement/feature "설명"   # 작업 시작 (session ID 자동 생성)
/task done "완료 요약"                  # 현재 작업 완료
/task log fix/bug "요약"               # 이미 끝난 작업 즉시 기록
/task fail "실패 사유"                  # 현재 작업 실패 처리
```

수동 흐름:
1. 작업 시작: `status: "in_progress"`로 항목 추가 (session ID 포함)
2. 작업 완료: `status: "done"`, `summary`에 결과 요약
3. 실패/중단: `status: "failed"`, `summary`에 사유

### Type 분류
| type | 용도 | 예시 |
|------|------|------|
| `experiment` | ML 학습/평가 | phase1/maml, baseline sanity check |
| `research` | 조사/분석 | 논문 리뷰, 데이터셋 분석 |
| `implement` | 기능 구현 | 새 모델, 대시보드 기능 |
| `fix` | 버그 수정 | 경로 오류, 학습 루프 버그 |
| `review` | 코드/결과 리뷰 | 실험 결과 분석, PR 리뷰 |
| `config` | 설정/인프라 | 디렉토리 구조, 환경설정 |

## 주요 참고 문서
- 연구 방향/실험 설계: `docs/research_and_experiments.md`
- Related works: `docs/relworks.md`
- 데이터셋 스펙: `README.md`

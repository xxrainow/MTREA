# MTREA 프로젝트 개요·개발 지침·진행 현황

작성 기준: 2026-10-06 (KST)  
저장소: https://github.com/xxrainow/MTREA  
확인한 main: `276bf77` (2026-10-04)  
대상: lab 담당자와 학습 담당자, 이후 개발 세션

이 문서는 대화에서 합의한 방향과 현재 체크아웃을 종합한 인계 문서다. 코드 존재, 사용자 보고에 따른 실행 성공, 앞으로의 계획을 구분한다. 이번 문서 작성에서 GPU·로봇 실행이나 테스트 재실행은 하지 않았다. 개발 규칙의 기준은 루트 `AGENTS.md`이며, 결정 변경 시 해당 파일도 함께 갱신한다.

## 1. 프로젝트 개요

**MTREA = Multi-Task Retentive Embodiment Adaptation**

초기 컨셉은 **소규모 로봇 팔 환경에서의 few-shot 크로스 임보디먼트 적응과 멀티태스크 능력 보존**이다. 사전학습된 VLA를 SO-101의 소량 시연으로 좁은 태스크에 적응시켰을 때, 학습한 태스크와 학습하지 않은 기존 태스크에서 어느 정도 성공하는지 연구한다.

현재 구축 중인 기반은 다음 전체 파이프라인이다.

```text
SO-101 연결·캘리브레이션
  → 텔레옵으로 시연 수집
  → 데이터 검증·필요한 통계 준비
  → GPU 서버에서 π0.5 fine-tuning
  → 체크포인트 로드·원격 추론
  → 실제 로봇 rollout
  → 성공·실패 기록 및 방법별 분석
```

브라우저 인터페이스는 이 과정을 실행하고 관찰하는 편의 도구다. 연구 실험은 UI 없이 터미널에서도 재현할 수 있어야 한다. LeLab은 인터페이스의 참고 사례이며, 현재 코드가 LeLab에 직접 의존한다는 뜻은 아니다.

### 연구 용어와 비교 기준

| 용어 | 의미 |
|---|---|
| T | 원래 embodiment에서 기반 VLA가 사전학습한 넓은 태스크 분포 |
| T_sub | SO-101 시연으로 실제 fine-tuning하는 태스크 부분집합 |
| held-out | T에서 T_sub를 제외한 태스크. 학습 데이터의 일반적인 test split과 다름 |
| H0 | naive single-task fine-tuning baseline. 현재 `train_expert_only=true` |
| H1, H2, … | H0와 비교할 다른 fine-tuning 방법 |
| episode | 사람이 텔레옵으로 기록한 시연 한 번 |
| rollout | 정책이 실제 로봇에서 태스크를 수행한 시도 한 번 |
| oracle | held-out 태스크 자체의 시연으로 학습한 별도 참고 모델. 해당 태스크의 참고 상한이며 비교 baseline이 아님 |

- Baseline은 H0다. SO-101 zero-shot 성공률은 embodiment mismatch만으로도 매우 낮을 수 있어 태스크별 비교 기준으로 삼지 않는다.
- 결과는 절대 성공률과 성공률 차이로 비교한다. near-zero baseline으로 나누는 보존율·성능 배수는 사용하지 않는다.
- oracle용 시연은 H0 및 다른 비교 방법의 학습 데이터에 들어가면 안 된다.
- 첫 자체 데이터 학습·pick & place 실행 성공은 파이프라인 검증이다. 정해진 설정과 평가 절차로 측정한 H0 연구 결과와 구분한다.
- SO-101에서 관측한 차이를 곧바로 순수한 망각량이라고 단정하지 않는다. embodiment 적응과 능력 보존을 함께 고려해 해석한다.

## 2. 두 사람의 역할 분담

| 영역 | 주 담당 | 책임 |
|---|---|---|
| `lab/core/`, `lab/server/`, `lab/web/` | 본인: lab 담당 | 로봇 작업의 인터페이스, 잡 관리, 원격 실행, API, UI |
| 최상위 `scripts/` | 본인: lab 담당 | 연구 실행 진입점을 호출하는 얇은 래퍼와 실행 계약 |
| `research/` | 친구: 학습 담당 | 학습 설정·방법, 데이터 학습 조건, 태스크 정의, 연구 평가·분석 |
| `robot/`, `docs/` | AGENTS.md상 학습 담당 소유 | 공유 로봇 설정과 연구 문서. lab 변경에 필요한 수정은 상호 리뷰 |
| 수집·학습·추론 통합 검증 | 공동 | 실제 데이터와 하드웨어로 양쪽 연결 확인 |

소유권은 수정 금지가 아니라 리뷰 책임을 뜻한다. 다른 사람 영역의 변경은 담당자가 리뷰하는 PR로 반영하고 main에 직접 push하지 않는다. 이 인계 문서 역시 docs 영역의 PR 리뷰 대상이다.

정책 서버의 모델 로딩·추론 구현은 학습 담당, 터널·클라이언트·로봇 실행은 lab 담당으로 나누는 것을 제안한다. 구체적인 서버 진입점과 통신 규약은 구현 전에 함께 확정해야 한다.

## 3. 현재 기술 결정과 이전 문서 정정

| 항목 | 현재 기준 |
|---|---|
| 학습 프레임워크 | LeRobot-native π0.5, PyTorch |
| 기반 모델 | `lerobot/pi05_base` |
| 학습 명령 | `lerobot-train --policy.type=pi05` |
| H0 | `train_expert_only=true`; π0.5에 LoRA를 사용하지 않음 |
| Python | 3.10 이상 |
| 하드웨어 | SO-101 leader/follower + 카메라 2대 |
| 장치 I/O·녹화·캘리브레이션 | LeRobot |
| 웹 백엔드 | FastAPI, 포트 8001 (구현 예정) |
| 프론트엔드 | React + Vite, 포트 8080 (구현 예정) |
| 학습 실행 위치 | SSH로 접근하는 GPU 머신 |
| 로봇 측 정책 접속 | `127.0.0.1:8765`, SSH 터널 입구 |
| GPU 측 정책 서버 | localhost에만 바인딩. 포트는 `remote.yaml`에서 설정; 이전 설계 예시는 8000 |
| 로봇 상수 | `robot/config.py`가 단일 출처 |
| 머신별 설정 | `robot/configs/*.yaml` |
| 셸 래퍼 | 최상위 `scripts/` |
| 잡 장부 | `data/jobs/jobs.jsonl`, append-only |
| run 산출물 | `data/runs/<run_id>/` |

9월 문서의 openpi/JAX, `third_party/openpi`, openpi `_CONFIGS` 등록, SO101Inputs/Outputs, openpi용 변환 단계는 현재 학습 경로의 전제가 아니다. 상수 위치를 `lab/config.py`로 적은 부분과 scripts 위치 미결도 현재 결정으로 대체한다.

정확한 LeRobot release tag와 환경 재현 방법은 아직 고정해야 한다. 현재 학습 스크립트는 v0.6.1을 기준으로 작성됐지만, 이것만으로 모든 머신의 설치 버전이 고정됐다고 볼 수 없다. uv/pip 선택도 미결이다.

### 데이터 변환과 정규화

현재 목표는 `convert`와 `norm_stats` 폴더를 무조건 만드는 것이 아니라, 학습기가 읽을 수 있는 데이터와 통계를 준비하는 것이다.

- 현재 학습 래퍼는 데이터셋의 `meta/stats.json`에서 `observation.state`, `action`의 `q01/q99` 존재를 검사한다.
- 필요한 통계가 없으면 원본을 보존한 사본에 재계산한다. 구체적인 명령 지원은 설치된 LeRobot 버전에서 확인한다.
- LeRobotDataset을 그대로 사용할 수 있으면 openpi 포맷 변환은 필요 없다.
- `data/norm_stats/`는 현재 폴더 규약에 남아 있지만, 별도 파일 생성이 현재 학습의 필수 완료 조건은 아니다.
- dry-run은 설정 확인이다. 실제 배치 로딩·GPU 학습·체크포인트 재로드 검증을 대신하지 않는다.

## 4. 저장소 구조

아래는 현재 파일과 예정 구조를 구분한 개요다. `(예정)`은 구현 완료를 뜻하지 않는다.

```text
MTREA/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── lab/
│   ├── core/
│   │   ├── models.py              # dataclass 기반 공용 모델
│   │   ├── paths.py               # data/ 경로 규약
│   │   ├── store.py               # append-only 잡 장부
│   │   ├── runner.py              # submit / refresh / stop / 로그
│   │   ├── executors/
│   │   │   ├── base.py
│   │   │   ├── local.py
│   │   │   └── ssh.py             # 예정
│   │   ├── robot.py               # 예정
│   │   ├── calibration.py         # 예정
│   │   ├── recorder.py            # 예정
│   │   ├── replayer.py            # 예정
│   │   └── policy_client.py       # 예정
│   ├── server/                   # 예정: main.py, ws.py, api/
│   └── web/                      # 예정: React + Vite
├── research/
│   ├── train.py
│   ├── test_train.py
│   ├── collect/
│   │   ├── tasks/*.yaml
│   │   └── protocol.md
│   └── analysis/
│       ├── metrics.py
│       └── test_metrics.py
├── scripts/
│   └── train.sh
├── robot/
│   ├── config.py
│   └── configs/                  # 장치·카메라·원격 머신 설정
├── experiments/
│   ├── record_test.sh
│   ├── inspect_dataset.py
│   └── gpu_smoke_test.sh
├── data/                         # gitignored
│   ├── raw/
│   ├── datasets/
│   ├── norm_stats/
│   ├── runs/
│   ├── rollouts/
│   ├── jobs/
│   └── logs/
├── docs/
└── tests/
    ├── test_core.py
    ├── test_runner.py
    └── test_boundary.py
```

추가 학습 방법, recipe, 평가 기록 진입점, 정책 서버와 `serve_policy.sh`는 실제 구현 필요에 맞춰 추가한다. 예정 폴더를 미리 만드는 것 자체가 목표는 아니다.

## 5. 개발 지침

### 계층과 의존성

- `lab/core/`: 로봇·잡·상태 로직. FastAPI나 HTTP에 의존하지 않는다.
- `lab/server/`: 요청 검증과 core 호출, 상태·로그 전달. API handler에 로봇 제어 루프를 넣지 않는다.
- `lab/web/`: 사용자 입력과 표시. Python·로봇·학습 로직을 넣지 않는다.
- `research/`는 `lab/`을 import하지 않는다.
- `lab/core/runner.py`는 `research/`를 import하지 않고 셸 래퍼를 로컬 또는 SSH subprocess로 실행한다.
- 공용 모델은 현재 dataclass다. HTTP 전용 스키마는 서버 계층에 둔다.
- lab과 research는 데이터 배치·설정·CLI 실행 규약으로 연결한다. 로봇 상수는 `robot/config.py`에서만 정의한다.

현재 `tests/test_boundary.py`는 research→lab 금지만 검사한다. core→FastAPI, runner→research 금지 규칙도 존재하지만 자동 검사 보강은 앞으로 할 일이다.

### 터미널 우선과 Fake의 범위

- 실제 동작은 터미널에서 성공시킨 뒤 core로 감싸고 UI에 연결한다.
- 집에서는 FakeBackend로 데이터 전달·렌더링·연결 상태 전환을 검증할 수 있다.
- Fake 관절값 화면은 ‘녹화 데이터 재생’으로 명시한 개발용 모니터다. 실제 Teleop 완료로 기록하지 않는다.
- 실물 Calibrate/Teleop/Collect 조작 UI는 해당 하드웨어 절차를 검증한 뒤 연결한다.
- Fake가 성공했다고 실물 연결·제어가 검증된 것으로 취급하지 않는다.

### 실험·기록

- seed를 반드시 명시한다.
- run마다 설정·실행 커밋·메트릭·체크포인트를 보존한다.
- 기존 run을 덮어쓰지 않는다.
- rollout 결과는 append-only다. 잘못된 결과를 정정할 때도 원기록을 삭제하지 않는 방식을 사용한다.
- 시연 episode 재녹화·삭제와 이미 수행한 평가 rollout 기록 변경을 구분한다.
- 태스크 역할 변경은 기존 정의 수정이 아니라 새 태스크 정의로 취급한다.
- 데이터셋과 태스크 역할의 연결을 명시해 held-out 시연이 비교 방법의 학습에 섞이지 않게 한다.
- 분석 코드는 합성 입력으로 GPU·로봇 없이 검증 가능해야 한다.

### 보안·협업

- `data/`, `.env`, calibration 파일은 커밋하지 않는다.
- 머신별 실제 연결 설정은 gitignore하고 공유용 example을 사용한다.
- 정책 서버를 `0.0.0.0`에 바인딩하지 않는다. localhost + SSH 터널 원칙을 유지한다.
- SSH 접속, 터널 연결, 정책 로드, 추론 준비 상태를 구분한다. 포트 연결만으로 정책 준비 완료를 표시하지 않는다.
- 기능을 작게 나눈 PR로 리뷰하고 main 직접 push를 하지 않는다.
- 코멘트·docstring은 영어로 간결하게 작성한다.
- 결정 변경 시 AGENTS.md와 관련 문서를 같은 변경에서 갱신한다.

## 6. 현재 태스크·수집 규약

현재 YAML에 이미 정해진 값은 다음과 같다. 아래 개수는 목표이며 수집·평가 완료 개수가 아니다.

| 태스크 | 역할 | 현재 목표 |
|---|---|---|
| pick_place | T_sub | 시연 50 episodes |
| push | held-out | 평가 30 rollouts |
| stack | held-out | 평가 30 rollouts |
| pour | held-out | 평가 30 rollouts |
| open_drawer | held-out | 평가 30 rollouts |

`research/collect/protocol.md`에는 태스크별 최소 3가지 색상·2가지 물체 유형, 특정 색상/유형 조합 편중 제한, 구체화한 지시문을 episode에 기록하는 규칙이 있다. held-out 평가 리셋은 기준 이미지 overlay를 사용하는 것으로 문서화돼 있다. overlay UI 구현은 아직 확인되지 않았다.

추가로 정할 항목은 T_sub 평가 횟수, 실행 제한 시간, 중단·장비 오류의 분모 처리, 반복 seed, few-shot 비교에 사용할 데모 수 단계 등이다. protocol의 ‘held-out은 학습용 시연을 수집하지 않는다’는 표현은 비교 방법에 적용하고, oracle용 별도 시연은 최신 AGENTS.md에 따라 분리한다. 관련 문구는 학습 담당 리뷰로 정합성을 맞춘다.

## 7. 지금까지 한 일

### 7.1 lab 기반 — 코드 구현 및 사용자 보고

- PR 1: core/server/web 구조, 설정 분리, AGENTS.md와 gitignore 정리.
- PR 2: `models.py`, `paths.py`, 기반 테스트.
- PR 3: `store.py`, `runner.py`, local executor.
- 잡 등록·상태 갱신·중지, 로그 읽기, 학습 중복 방지 로직 구현.
- append-only `jobs.jsonl`로 실행 이력 보존.
- 사용자 보고: 당시 테스트 12개 통과, 터미널에서 프로세스 실행·중지 확인, 서버 재시작과 별개로 로컬 잡이 유지되는 구조 확인.
- 이 테스트 개수는 당시 보고이며 현재 전체 테스트 개수를 의미하지 않는다.

### 7.2 로봇 데이터 수집 — 사용자 보고

- `experiments/record_test.sh`로 녹화 명령 준비.
- 학교에서 포트·카메라 설정을 YAML에 기록.
- 실제 SO-101에서 `lerobot-record`로 데이터 수집 성공, raw 데이터 보관.
- `experiments/inspect_dataset.py`로 수집 데이터 검사 수행.

**검증 범위 주의:** 현재 체크아웃의 inspector는 메타데이터, 관절·카메라·FPS 및 첫 parquet 중심 확인이다. 전체 데이터의 dtype·지시문·NaN/Inf 검사는 구현돼 있다고 확인할 수 없다. 학교에서 사용한 버전이 다르면 그 파일과 결과를 대조하고, 전체 학습 입력 검사는 보완해야 한다.

실제 데이터 경로도 명확히 기록해야 한다. 저장소의 상대 경로 `data/raw/`와 시스템 루트의 `/data/raw/`는 서로 다르다.

### 7.3 학습·연구 — 코드 확인

| 시점 | 변경 | 상태 |
|---|---|---|
| 기존 연구 코드 | 태스크 YAML, 수집 프로토콜, 성공률·방법 비교·데이터 효율 분석 함수 및 테스트 | 코드 존재 |
| 9/30, PR #7 | 연구 질문·oracle 규칙 보완 | main 반영 |
| 10/3, PR #8 | `experiments/gpu_smoke_test.sh` | main 반영; 실제 GPU 결과는 별도 확인 필요 |
| 10/4, PR #9 | `research/train.py`, `research/test_train.py`, `scripts/train.sh` | main 반영 |
| 10/4, `276bf77` | 실패 시 로그 마지막 30줄 출력, Python 출력 버퍼링 방지 | main 반영 |

GPU smoke test는 학습 설정·배치 크기별 메모리와 속도를 확인하는 도구다. 체크포인트를 저장하지 않으므로 체크포인트 재로드 검증을 대신하지 않는다.

학습 래퍼는 다음을 제공한다.

- H0 설정과 명시적인 seed.
- `q01/q99` 존재 검사와 `--dry-run`.
- 기존 run 덮어쓰기 방지.
- `config.json`: 인자·LeRobot 명령·git commit/dirty·버전·호스트·시작 시간.
- `train.log`: 전체 학습 로그.
- `metrics.jsonl`: step/loss/learning rate.
- `result.json`: 완료·실패·중단 결과.
- runner용 `STATUS`와 `PROGRESS` 출력.
- 중단 신호 전달 및 실패 로그 표시.

현재 실제 LeRobot 출력 경로는 `data/runs/<run_id>/lerobot/`이며 체크포인트는 이 하위에 생성된다. UI는 `data/runs/<run_id>/checkpoints/`라고 단정하지 말고 실제 산출물 경로를 따른다.

### 7.4 아직 완료되지 않은 부분

- SSH executor·터널 관리.
- robot/calibration/recorder/replayer/policy_client core 구현.
- FastAPI·WebSocket·React 화면.
- 자체 수집 데이터의 GPU 학습 성공 및 체크포인트 재로드 확인.
- π0.5 정책 서버와 lab 사이의 추론 통신 연결.
- 실제 정책 rollout 및 결과 저장 흐름.
- Collect·Inference·Eval 화면.

문서 작성 시 inspector와 로컬 장치 YAML에 미커밋 수정이 있었다. 이를 원격에 반영된 완료 내역과 혼동하지 않는다.

## 8. 앞으로의 개발 순서

집에서는 인터페이스 기반을 만들고 학교·GPU 접근이 가능해지면 실물 검증을 앞당긴다. 아래 순서는 모든 단계가 끝나야 다음 사람이 작업할 수 있다는 뜻이 아니다.

| 단계 | 작업 | 주 담당 | 완료 기준 |
|---|---|---|---|
| 1 | 화요일 수집 데이터 검사·필요한 통계 재계산·학습 dry-run | 공동 | 원본 보존, 학습 입력 검사 결과와 통계 확인, 의도한 실행 설정 출력 |
| 2 | `robot.py` 최소 인터페이스 + FakeBackend | lab | 연결·해제·상태 읽기·녹화값 재생. 미구현 실제 backend는 명확히 사용 불가 처리 |
| 3 | FastAPI `main.py` | lab | 8001에서 health/status/jobs 조회, core 위임 확인 |
| 4 | React/Vite 기본 화면 | lab | 8080에서 사이드바·상태 표시, HTTP 조회와 WS 이벤트 송수신 검증 |
| 5 | Fake 관절값 개발용 모니터 | lab | 6개 관절값 표시, Fake/녹화 재생 상태 명시. 실제 Teleop 완료로 간주하지 않음 |
| 6 | Datasets 페이지 | lab | 실제 데이터셋 목록·episode 수·태스크 역할·검사/통계 상태 표시 |
| 7 | Train 페이지 | lab | 기존 H0 설정으로 작업 요청, 잡 목록·로그·중지. 로컬 테스트 작업으로 UI 동작 검증 |
| 8 | 학교: LeRobotBackend 실물 연결·추가 녹화 | lab 중심 공동 | 기존 녹화 재현, 관절·카메라 읽기, 텔레옵 시작·정지·오류 정리 확인 후 UI 연결 |
| 9 | SSH executor·터널·기존 train.sh 연결 | lab + 학습 | 원격 일반 작업 검증 후 자체 데이터 짧은 학습, 산출물·종료 상태 확인 |
| 10 | 정책 서버 + policy_client + Inference | 공동 | 체크포인트 재로드, 저장 관측값 추론, 실제 팔 실행, 최소 rollout 결과 저장 |
| 11 | Collect 페이지 | lab | 검증한 녹화 절차의 시작·정지·저장·재녹화 UI. 터미널 수집이 충분하면 후순위 가능 |
| 12 | Eval 대시보드 | lab + 학습 | 누적 rollout으로 태스크별 절대 성공률, H0 대비 차이, 별도 oracle 참고값 표시 |

### 단계별 구현 범위

**Datasets:** dataset ID/경로, episode·frame 수, FPS, 카메라·관절 구성, 태스크 역할, 통계 유효성, 검사 결과·시점을 표시한다. `norm_stats` 폴더 존재만으로 준비 완료를 판단하지 않는다. raw 데이터와 학습용 데이터의 관계도 기록한다.

**Train:** 처음에는 H0 하나만 제공한다. 데이터셋·seed·run 경로를 명시하고, 아직 구현되지 않은 H1/H2 recipe를 선택지로 노출하지 않는다. `train.sh`를 다시 만들지 않고 기존 래퍼를 연결한다.

**SSH:** GPU를 쓰지 않는 짧은 원격 작업으로 실행·로그·종료 코드·중지를 먼저 검증한다. 네트워크 단절·재접속 후 상태 확인과 서버 재시작 후 원격 작업 추적을 검증한다. 계정 대기 중에는 API/UI 개발을 진행할 수 있다.

**Inference:** 터널 연결과 정책 준비 상태를 분리한다. 모델 입력·출력, 카메라 매핑, 관절 순서, 정규화 처리 위치, action chunk 처리, timeout·중단 시 동작을 공동으로 정의한다.

**최소 rollout 기록:** Eval 화면이 없어도 첫 실제 정책 실행부터 필요하다. 권장 필드는 rollout ID, task ID/정의 버전, 방법, run·checkpoint, 실행 시각, 결과(success/failure/aborted), 사유·메모다. 학습 run을 통해 seed·커밋·데이터셋을 추적할 수 있어야 한다. 중단 시도의 통계 포함 규칙은 실험 전에 확정한다.

**Collect:** 처음부터 `lerobot-record`의 모든 옵션을 UI로 옮기지 않는다. 이미 성공한 절차부터 감싸고 녹화 프로세스 상태와 실제 저장 완료를 구분한다.

**추가 화면:** Calibrate와 실물 Teleop은 하드웨어 절차 검증 후 구현한다. Checkpoints는 Train/Inference의 선택 기능으로 시작하고 목록이 복잡해질 때 별도 페이지로 분리할 수 있다. Replayer는 필요 시 추가한다.

## 9. 함께 확정하거나 검증할 사항

### 다음 통합 전에 필요

- [ ] 실제 학습에 사용할 LeRobot release tag, Python/PyTorch/CUDA 환경과 설치 절차 고정.
- [ ] 학교 수집 데이터의 실제 위치·버전·episode 수와 inspector 버전 확인.
- [ ] 전체 데이터 dtype·지시문·NaN/Inf 및 실제 학습 로딩 검증 보완.
- [ ] 데이터셋 식별·버전·로컬→GPU 전송 방법과 경로 규약 정의.
- [ ] GPU 실행 디렉토리·환경 활성화·로그·산출물 경로 확정.
- [ ] 정책 서버 구현 담당과 실행 명령·통신 규약 확정.
- [ ] core→FastAPI 및 runner→research 경계 테스트 보강.
- [ ] T_sub 평가 횟수·제한 시간·중단 처리·seed·few-shot 데모 수 단계 결정.
- [ ] oracle 관련 문서 표현과 최신 AGENTS.md 정합성 맞추기.

### 뒤로 미뤄도 되는 선택

- zustand 등 프론트 상태관리와 shadcn 등 UI 킷.
- JSONL 장부의 SQLite 전환.
- SO-101 외 로봇까지 지원하는 범용 backend 추상화.
- 사용하지 않는 recipe·변환 단계·독립 norm_stats 작업 UI.

카메라 이름·FPS·관절 순서는 고정 상수, 장치 인덱스 등 머신별 값은 YAML이라는 원칙을 유지한다. 현재 CAMERAS의 인덱스·해상도 배치를 더 정리할지는 robot 담당자와 리뷰한다.

## 10. 개발·검증 명령 참고

```bash
# 기본 개발 환경 및 CPU 측 테스트
pip install -e .
pytest research/ -v
pytest tests/test_boundary.py
pytest tests/test_core.py tests/test_runner.py

# 설정 확인 예시: repo ID와 경로는 실제 값으로 교체
# dry-run은 학습을 실행하지 않으며, 통계가 미확인이어도 출력이 가능하므로 메시지를 확인한다.
bash scripts/train.sh \
  --method=h0 \
  --dataset=<repo_id> \
  --dataset-root=<local_dataset_root> \
  --output=data/runs/h0_pickplace_seed0 \
  --seed=0 \
  --dry-run
```

GPU 학습·정책 서버·로봇 명령은 해당 환경을 확인한 뒤 실행한다. 예시의 placeholder를 그대로 실행하지 않는다. 커밋되지 않은 변경으로 실험할 경우 commit hash만으로 재현이 충분하지 않으므로 사용한 변경도 추적 가능하게 남긴다.

## 11. 단계별 최종 목표

1. **데이터 준비 완료:** 수집 데이터와 학습 입력 검사가 일치하고 필요한 통계가 준비됨.
2. **학습 경로 완료:** 자체 데이터로 학습하고 저장한 checkpoint를 다시 로드함.
3. **인터페이스 연결 완료:** UI에서 잡·데이터·로그를 확인하고 승인된 작업을 실행·중지함.
4. **로봇 실행 경로 완료:** 정책이 실제 SO-101에서 동작하고 모든 평가 시도가 기록됨.
5. **H0 평가 완료:** 고정한 프로토콜로 T_sub와 held-out 성공률을 측정함.
6. **연구 비교:** 동일한 held-out 조건에서 H0와 다른 방법을 비교하고 oracle은 별도 참고값으로 표시함.

## 12. 참고 자료

- 현재 개발 규칙: `AGENTS.md`
- 연구 정의: `docs/PROBLEM.md`
- 관련 연구: `docs/RELATED_WORK.md`
- 수집 규약: `research/collect/protocol.md`
- 태스크 정의: `research/collect/tasks/*.yaml`
- 학습 진입점: `research/train.py`, `scripts/train.sh`
- [PR #7: 연구 정의](https://github.com/xxrainow/MTREA/pull/7)
- [PR #8: GPU smoke test](https://github.com/xxrainow/MTREA/pull/8)
- [PR #9: 학습 래퍼](https://github.com/xxrainow/MTREA/pull/9)
- [이전 프로젝트 개요](https://claude.ai/code/artifact/67bfbe76-3450-4b6f-ae9c-5be9241225a6)
- [이전 개발 지침](https://claude.ai/artifact/MKj9d7bWbqEkHVNevsgP8h)

이전 Claude 문서는 도구로 본문에 접근하지 못했으며, 사용자가 대화에 제공한 내용을 참고했다. 충돌하는 openpi 관련 내용은 현재 LeRobot-native 결정으로 대체한다.

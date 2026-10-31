# Task 11 — Train / Checkpoint 구현 분석

## 결과와 범위

**승인된 데모를 `feat/train`의 React 화면에 반영했다.** Train의 설정/모니터 탭을 제거하고 Checkpoint를 독립 메뉴로 추가했다.

- 설정 → 진행 → 완료 모니터를 유지한다. 체크포인트 상세와 목록은 사용자가 버튼으로 연다.
- Start training은 Collect의 Start recording처럼 제목 오른쪽에 둔다.
- Compute target을 먼저 선택하고 학습 입력·학습 조건을 2열로 배치한다.
- Local은 브라우저 기기가 아니라 웹 백엔드 실행 머신을 뜻한다.
- 화면 문구는 기존 앱에 맞춰 영어로 작성했다.

**현재 실행은 명시적 시뮬레이션이다.** 학습 API와 SSH executor가 없으므로 실제 작업을 실행하지 않는다. 입력값 검증과 서버 검사를 구분한다. 서버 검사를 통과했다고 표시하지 않는다. 체크포인트는 메모리의 메타데이터이며 모델 파일이 아니다.

## 파일별 변경

| 파일 | 구분 | 핵심 변경 |
| --- | --- | --- |
| `src/training/model.ts` | 생성 | 설정·run·checkpoint 타입, 기본값, 검증, 순수 reducer |
| `src/pages/TrainPage.tsx` | 수정 | 설정/진행 분기, 상단 시작 버튼, Compute target, 고급 조건, 중지 확인 |
| `src/pages/CheckpointPage.tsx` | 생성 | checkpoint ID 기반 상세/목록, 빈 상태, 설정 조회 |
| `src/App.tsx` | 수정 | 앱 수준 reducer 소유, Checkpoint 라우팅, 완료 모니터 유지·명시적 상세 열기 |
| `src/components/AppLayout.tsx` | 수정 | Checkpoint 메뉴와 PageName 확장 |
| `src/components/AppLayout.css` | 수정 | 모바일에서 사이드바의 sticky/전체 화면 높이 해제 |
| `src/pages/TrainPage.css` | 수정 | 기존 탭·모니터 전용 스타일 제거, 헤더 액션·진행률·상세/목록·반응형 스타일 |
| `src/pages/OverviewPage.tsx` | 수정 | 잘못된 H0 expert-only 표기를 Full fine-tuning으로 수정 |
| `tests/training.test.mjs` | 생성 | 모델 상태 전이와 실험 설정의 경계 조건 테스트 8개 |
| `TRAIN_IMPLEMENTATION.md` | 생성 | 구현 범위·개발 흐름·검증·면접 포인트 기록 |
| 저장소 루트 `AGENTS.md` | 수정 | 화면 구조, 시뮬레이션 범위, 상태 보존 범위 기록 |

## 주요 코드

### model.ts

**`TrainingConfig`는 폼 입력을 문자열로 유지한다.** 숫자 필드가 비어 있는 상태와 0을 구분한다. 특히 Seed를 암묵적으로 0으로 바꾸지 않는다.

**`validateConfig()`는 실행 가능한 데모 설정인지 검증한다.** Seed·정수 필드·유한 양수 LR·warmup < steps·run 이름을 확인한다. SSH 레시피는 batch 16 이하로 제한한다. 이미 결과가 있는 run 이름을 재사용하지 못하게 한다. 이는 실제 서버 디렉터리 중복 검사가 아니다.

**`trainingReducer()`가 상태 전이를 한 곳에서 통제한다.**

- `configure`: 진행 중 변경 차단. 그 외 변경은 검증 결과 무효화.
- `check`: 입력 오류 계산. 유효할 때만 시작 가능 상태로 전환.
- `start`: 검증 여부·활성 작업·현재 입력을 다시 확인. 설정을 복사해 run에 고정.
- `tick`: App의 타이머에서 매초 진행. 합성 loss/LR·로그를 누적하고 지나간 모든 저장 간격의 스냅샷 생성.
- `complete`: 최종 step의 스냅샷을 추가. 중복 완료는 무시.
- `disconnect` / `reconnect`: 같은 run ID를 유지. 연결 끊김을 종료로 취급하지 않음.
- `stop`: 기존 스냅샷 보존. 현재 step의 저장을 보장하지 않음.
- `new`: 활성 run이 없을 때 설정 화면으로 복귀. 과거 스냅샷 보존.

**`isTrainingActive()`는 running과 disconnected를 모두 활성 상태로 취급한다.** 관측이 끊긴 작업을 없는 작업으로 오인해 두 번째 작업을 시작하지 않도록 한다.

### TrainPage.tsx

**`TrainPage()`는 상태를 표시하고 의도를 dispatch한다.** 페이지 내부에 run을 보관하지 않는다. 중지 확인 UI만 로컬 상태로 둔다.

- `configure()`: 부분 입력을 합쳐 configure 액션 전송.
- `numericField()`: 일관된 label/input 구조와 숫자 입력 생성.
- 자동 완료: App 타이머는 완료 상태에서 정지한다. 최종 그래프·로그를 유지하며 New training setup을 눌러야 설정으로 돌아간다.
- `crypto.randomUUID()`: 이벤트 핸들러에서 run ID 생성. reducer 안에서 난수를 생성하지 않아 순수성을 유지.

**그래프 값은 실측이 아닌 합성값임을 명시한다.** loss는 데모 곡선, LR은 warmup/cosine 형태다. 로그도 SIMULATION으로 구분한다. 실제 학습 지표로 해석하면 안 된다.

### App.tsx / CheckpointPage.tsx

**`App()`이 상태 수명을 결정한다.** `useReducer()`를 페이지보다 위에 두므로 Train 컴포넌트가 언마운트되어도 run과 checkpoint가 남는다. 새로고침하면 초기화되며 영속성은 구현하지 않았다.

**`openCheckpoint()`는 이동 대상 ID를 명시한다.** 완료 직후 특정 결과를 여는 경우 ID를 전달한다. 사이드바의 `navigate()`는 선택 ID를 지워 항상 목록부터 연다.

**`CheckpointPage()`는 ID로 상세를 선택한다.** 선택된 결과가 없으면 목록/빈 상태를 표시한다. 목록은 최신 생성 항목부터 보여준다. 각 행은 run 이름·step·target·시뮬레이션 상태를 제공한다.

## 개발 흐름

1. **기존 구조를 확인했다.** `feat/train`의 깨끗한 작업 트리, 페이지 조건부 렌더링, 학습 wrapper 기본값, SSH executor 부재를 확인했다.
2. **UI 계약을 먼저 정했다.** 네 화면과 상단 액션을 확정하고 실제 실행과 시뮬레이션의 표시를 구분했다.
3. **상태 모델을 분리했다.** 설정·run·checkpoint를 정의하고 검증과 전이를 순수 함수로 옮겼다.
4. **메뉴와 화면을 연결했다.** App에 상태를 올리고 완료 모니터 유지와 상세 버튼, 사이드바 목록 진입을 연결했다.
5. **실험 설정을 맞췄다.** H0 full fine-tuning, steps 10000, batch 16, peak LR 2.5e-5, 명시적 Seed를 적용했다.
6. **경계 조건을 검증했다.** 중복 실행·검증 무효화·연결 끊김·중지·중복 checkpoint 생성 방지를 테스트했다.
7. **브라우저로 흐름을 확인했다.** 설정부터 목록까지 직접 조작하고 좁은 화면의 사이드바·테이블 가독성을 보완했다.

## 검증

- `npm run build --prefix lab/web`: TypeScript와 Vite 빌드 통과.
- `npm run lint --prefix lab/web`: ESLint 통과.
- `node --test lab/web/tests/training.test.mjs`: 8개 통과. TypeScript로 모델을 변환한 후 Node 내장 테스트 실행기를 사용한다.
- `pytest tests/test_boundary.py -q`: 3개 통과. research/lab 의존성 경계 유지.
- 브라우저: Seed와 데이터셋 입력, 검증 후 시작 활성화, 진행률 변경, Checkpoint 이동 후 Train 상태 유지, 당시 완료 후 최종 상세 자동 이동과 목록 복귀 확인. 이후 요구 변경으로 자동 이동을 제거했다.
- 하드웨어·CUDA·SSH·실제 학습은 테스트하지 않았다. 이 변경은 해당 기능을 구현하지 않는다.

## 실제 원격 실행으로 확장할 때

**UI 검증을 서버 권위로 대체해야 한다.** 서버는 dataset role·학습 호환성·q01/q99·출력 충돌·GPU 할당·인증을 다시 검사해야 한다. 화면의 disabled만으로 동시 실행을 막을 수 없다.

**실행 요청과 실행 확인을 분리해야 한다.** `starting`, `stopping`, `verifying`, `failed` 등 서버 상태를 추가하고, 요청 성공만으로 학습 시작/종료/저장을 확정하지 않는다.

**작업 ID와 저장된 결과가 복구의 기준이어야 한다.** SSH 세션과 독립적인 원격 프로세스, 재접속 조회, 영속 job store, 최종 파일 확인을 구현한다. 연결 단절 후 자동 재실행하지 않는다.

**연결 지점은 기존 계층을 따른다.** 브라우저 → FastAPI → core Runner → SSH executor → scripts/train.sh. core에서 research를 import하지 않는다. 사용자 문자열을 쉘 명령으로 직접 연결하지 않는다. 커맨드 인자 검증·quoting·서버 측 중복 방지가 필요하다.

**연구 결과에는 provenance가 필요하다.** 실행 호스트 commit, 확정 config, dataset identity, seed, 실제 checkpoint 경로를 저장해야 한다. 시뮬레이션에서 gitCommit은 null이며 실제 기록인 척하지 않는다.

## 면접 포인트

**1. 컴포넌트 수명과 작업 수명을 분리했다.** 페이지 로컬 상태로 run을 관리하면 메뉴 이동 때 사라진다. App reducer로 올려 해결했다. 브라우저 새로고침 복구까지 해결한 것은 아니며, 그 단계에는 서버 영속성이 필요하다고 설명한다.

**2. 상태 머신으로 불가능한 동작을 차단했다.** 활성 run의 중복 시작·설정 수정, disconnected 상태의 완료·중지를 reducer에서 거부한다. 버튼 비활성화와 모델 검증을 함께 둔 이유를 설명할 수 있다.

**3. 네트워크 단절과 작업 실패를 구분했다.** SSH가 끊겨도 학습 프로세스는 살아 있을 수 있다. unknown 상태를 유지하고 같은 run을 조회해야 중복 실행과 자원 충돌을 피한다. 현재는 그 UX/상태 정책만 시뮬레이션했다.

**4. 검증 결과의 유효 기간을 관리했다.** 값이나 compute target이 바뀌면 기존 검증을 무효화한다. 실서비스에서는 검증한 config revision과 실행 요청을 묶고 서버에서 재검증해야 한다는 확장점을 설명한다.

**5. 재현성 요구를 제품 UI에 반영했다.** 명시적 Seed, H0 정의, 고정한 run 설정, checkpoint와 원본 run 연결을 설명한다. 같은 seed가 완전한 GPU 결정성을 보장한다고 주장하지 않는다.

**6. 순수 함수 테스트와 브라우저 검증을 나눴다.** reducer 테스트는 상태 경계와 불변성을 확인한다. 브라우저 검증은 메뉴·포커스·화면 이동·레이아웃처럼 함수 테스트로 확인할 수 없는 부분을 보완한다.

**7. 구현한 범위를 정확히 말한다.** “GPU 학습 시스템을 완성했다”가 아니라 “원격 학습을 위한 프런트엔드 상태 모델과 체크포인트 탐색 UX를 구현·검증했다”가 이번 변경에 맞는 설명이다.

## 자동 모니터링·Jobs 추가 변경

- `src/components/TrainingMetricChart.tsx` 생성: Loss/LR SVG 그래프, 대기 상태, step 축, 최신 값, 테마 색상.
- `App.tsx`: 타이머를 페이지보다 위에 배치. 메뉴 이동 시 학습을 유지하고 타이머 정리로 중복 갱신 방지. 완료된 Train 재진입도 모니터를 유지한다. New training setup만 새 설정으로 전환한다.
- `model.ts`: `tick`, 합성 metrics, 최근 200줄 로그, 최근 500개 metric, run 이력 추가. checkpoint는 저장 간격을 따라 생성하며 결과와 job은 동일 run ID로 연결.
- `TrainPage.tsx`: 수동 Advance/Complete 버튼 제거. progress → Loss/LR 그래프 → 로그 순서.
- `JobsPage.tsx`: 기존 실제 `/api/jobs` 영역과 별도로 브라우저 시뮬레이션 job 이력 및 그래프·로그 제공. 실제 API에는 데모를 전송하지 않음.
- `tests/training.test.mjs`: 자동 완료·저장 간격·LR 최종값·job 이력 보존 검증 추가.
- 기본 시뮬레이션은 약 100 tick에 완료. 브라우저 백그라운드 탭에서는 타이머가 지연될 수 있다. 실제 장기 실행의 영속성은 서버 job store가 담당해야 한다.
- 연결 끊김 시 데모 tick은 중단한다. 실제 원격 프로세스 중지를 의미하지 않으며 재연결 시 상태를 조회해야 한다.
- 실제 학습 job은 실행 상태·로그를 관리하고 Checkpoint는 그 job의 산출물이다. 둘은 동일 데이터의 중복 페이지가 아니다.

## Run별 체크포인트 탐색

- `CheckpointPage.tsx`: 왼쪽 학습 Run 목록과 오른쪽 checkpoint 목록/상세로 재구성. `selectRun()`, `showDetails()`, `showList()`로 선택 흐름을 구분한다.
- `App.tsx`: checkpoint뿐 아니라 `training.runs`도 전달한다. 저장 당시 snapshot의 상태 대신 현재 run 상태를 표시하며 아직 checkpoint가 없는 run도 탐색할 수 있다.
- `CheckpointPage.css` 생성: 2열 레이아웃, 선택 강조, 공통 요약, 모바일 세로 배치. 표에서 중복 Run/Target 열 제거.
- Train에서 특정 checkpoint를 열면 해당 run을 우선 선택한다. 상세에서 목록으로 돌아와도 같은 run을 유지한다. 새 checkpoint가 추가돼도 사용자가 선택한 다른 run을 변경하지 않는다.
- 그룹화는 이름이 아니라 run ID 기준이며 체크포인트는 step 내림차순으로 표시한다.

# 002 — LeWM×f1tenth 계획 v2 (001 critique 반영)

> 2026-06-13. 초안(분석→결정→구현 3-Phase)에 대한 001 비판 검토를 전면 수용해 수정한 계획.
> 001의 Critical 3건은 본 세션에서 코드로 교차 검증 완료:
> - EMA 없음 — le-wm 전체에 ema 코드 0건, collapse 방지 = SIGReg(module.py:10)
> - cost = goal 임베딩 MSE뿐 — jepa.py:112-126 criterion이 마지막 step 임베딩 vs goal MSE가 전부
> - 렌더 부재 + venv 분리 확정 — vendored f110_env.py pyglet/render() 전부 주석(41-43, 339행),
>   기존 venv Python 3.8.10 vs le-wm 요구 3.10(README:34-36)
> append-only.

---

## 프로젝트 목표

f1tenth 시뮬레이터에서 LeWM(LeWorldModel, JEPA 기반 world model)을 offline 데이터셋으로
학습시키고, 학습된 world model로 MPC planning 주행 평가까지 수행한다.

- 데이터셋 생성원: `~/f1tenth_RL_project`의 DreamerV3 학습 policy snapshot
  (충족 확인됨: stage1_map_easy3 diversity bin 9개 6.1~18.4s + best 6.1s, stage2_oschersleben 16.6s)
- 정본 코드: `~/le-wm` (본체 815줄, 데이터 로딩은 외부 패키지 `stable-worldmodel`(swm) 의존)
- 선행 분석: `f1tenth_RL_project/_thinking/implementation/021` §2 (LeWM data contract)

---

## Phase 1 — 디리스킹 + LeWM 코드 분석

### Step 0 (최우선, SPOF 해소)
1. py3.10 venv 생성 → `stable-worldmodel[train,env]` 설치
2. 공식 pusht .h5 다운로드(HF, README:41-52)
3. **train.py 레퍼런스 런 1회**(smoke) — 코드/환경이 실제로 도는지 가장 싼 검증
4. 실물 공식 h5에서 스키마 역추출(h5dump) — swm 소스 분석보다 신뢰도 높은 SSOT

데이터 생성(py3.8, 기존 venv) ↔ 학습(py3.10, 신규 venv)에서 **.h5가 두 환경의 경계 인터페이스**.

### Step 0b (렌더 스파이크)
pyglet 복원이 아니라 **map occupancy + (x,y,θ) pose → 224×224 top-down 래스터 합성**
(numpy/PIL, 결정적·headless-safe) 프로토타입. 기존 eval npz의 pose 몇 개로 이미지 생성 확인.
이 결과가 Phase 2 결정①의 진짜 판단 재료 (반나절 분량).

### 코드 분석 (021 §2 검증 + 보강)
- jepa.py/module.py: encoder(ViT-tiny, embed 192)/predictor 구조,
  **loss = pred MSE + SIGReg (EMA 없음, target detach 없는 동일 encoder)**
- **정규화 contract**: pixels=ImageNet 정규화(utils.py:6-10), action 포함 non-pixels 전부
  z-score(train.py:59-66), 시퀀스 경계 NaN padding(train.py:25 nan_to_num), eval 시
  StandardScaler 재-fit(eval.py:71-82) → 데이터셋 스펙에 명시
- train.py/config: num_steps=4(history 3+pred 1)·frameskip=5 소비 방식, hydra 체계,
  `${eval:}` resolver(Step 0 smoke에서 자연 검증)
- eval.py: **단기 goal-reaching 전용 구조**임을 전제로 파악 — solver+get_cost 재사용 가능 /
  env 루프·goal 공급·action 역정규화는 f1tenth용 신규 구현 대상
- train/val 분리는 코드 내장(train.py:73-79, 0.9 random_split)이나 **window 단위**라
  val loss 낙관적 — 게이트로 쓸 때 인지

산출물: data contract 확정판(실물 h5 기준) + 렌더 스파이크 결과 + 모달리티 결정 판단 재료

---

## Phase 2 — 설계 결정 4개

### 결정① 관측 모달리티
- (a) top-down 래스터 이미지 수집: 정본 무수정. 실행 가능성은 Step 0b 스파이크로 판명
- (b) encoder lidar 1D 변형: **encoder 교체가 아니라 train/eval/swm 파이프 연쇄 수정**
  (encode가 HF ViT API 결합 jepa.py:37-38, 전처리 'pixels' 키 고정 train.py:59,
  eval 'pixels'/'goal' 고정 eval.py:61-64, goal 인코딩도 lidar화 필요)
- 스파이크 성공 시 (a)가 사실상 우세해지는 구조 — 스파이크 먼저, 결정은 그 후

### 결정② 데이터셋 구성 (v1 축소)
- map_easy3 단일 트랙, best + 중간 bin 2~3개로 시작 후 반복 조정. 트랙 2종/정교한 믹스는 v2
- policy는 lidar+state를 소비 → **"vector obs로 policy 구동 + 이미지 병행 기록" 이중 구조**
- dtype/압축은 공식 pusht h5 모방. 용량 ~17GB급(224×224×3, 300ep×400step) 감안

### 결정③ 평가 설계 (분기 + milestone 계단화)
- LeWM planning 스택은 "goal 이미지와의 임베딩 MSE"가 cost의 전부인 단기 goal-reaching 전용.
  "progress cost"는 코드에 없는 신규 시스템임을 전제로:
  - **(③-a) 레퍼런스 lap(best policy 에피소드)에서 N step 앞 프레임을 subgoal로 공급하는
    receding-horizon goal-chasing — 정본 get_cost 유지 (권장)**
  - (③-b) custom cost head 학습 (③-a 실패 시 fallback)
- milestone: 1차 = 단기 goal-reaching 성공률(정본 평가 방식 그대로) → 2차 = 완주/lap time

### 결정④ 시간 해상도 (신설)
- env step 0.02s(action_repeat=2) 기준, frameskip × CEM horizon × receding horizon ×
  subgoal 간격을 **한 묶음으로** 결정
- 기본값 유지 시(frameskip 5, horizon 5, action_block 5) lookahead 0.5초 —
  lap 6~13s 주행에 충분한지 검토

산출물: 결정 4개가 기록된 설계서 (필요 시 critic 재검토 사이클)

---

## Phase 3 — 구현

1. **데이터셋 파이프라인**: snapshot policy(lidar 구동) rollout + 이미지 병행 기록 →
   공식 h5 스키마/dtype/압축 모방 .h5 변환 → **swm 로더 통과 검증** (게이트)
2. **LeWM 어댑터**: f1tenth용 data yaml, action_dim=2 배선, 결정④ 반영
3. **학습**: smoke(소량 1회 통과) → 본 학습 → 모니터링
4. **평가 (실질 최대 공수)**: f1tenth용 planning 루프 신규 구현
   - 재사용: solver(CEM/Adam) + jepa.get_cost
   - 신규: env 루프, subgoal 공급(③-a), action 역정규화(누락 = 전형적 silent failure)
   - 지표: goal-reaching 성공률 → 완주/lap time → Dreamer 대비

각 단계 검증 게이트: 로더 통과 / smoke / dry-run 통과 후 다음 단계 진입.

---

## 미검증 항목 (Phase 1에서 해소 예정)

- swm 내부: load_dataset 윈도잉/frameskip 소비, h5 정확한 키·shape, NaN padding 규약,
  WorldModelPolicy/CEMSolver 정규화 처리, swm.World API → Step 0이 해소 장치
- LeWM 논문(arXiv 2603.19312): frameskip 의미, 권장 데이터 규모
- uv/py3.10 설치 가능 여부, GPU VRAM 여유

## 참조 문서 (필요 시점에 선별 읽기)

- `f1tenth_RL_project/_thinking/analysis/001-env-analysis.md` — 환경 파라미터·좌표계·함정
  (단, 일부 이슈는 이후 wrapper에서 해결됨 — understand/002, implementation/005~007과 함께)
- `implementation/021` §2 — LeWM contract 출발점
- `implementation/022, 029, 030` — snapshot/eval_gate 정책
- `planning/011~015` — heldout 프로토콜
- `env_setting/004~006` — venv 구성 참조

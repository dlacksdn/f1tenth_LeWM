# 001 — LeWM×f1tenth 계획서 비판적 검토 (critic 리뷰)

> 2026-06-13. Phase 1(코드 분석) → Phase 2(설계 결정 3개) → Phase 3(구현) 계획 초안에 대한
> 읽기 전용 검토 기록. 검증 대상: `~/le-wm` 전체 소스(815줄) + config 전부 + README,
> `f1tenth_RL_project/_thinking/implementation/021` §2, RL_project venv/snapshot/env 코드.
> append-only.

---

## 0. 종합 평결: **수정 후 진행**

골격(분석→결정→구현)과 021 §2의 data contract 분석은 코드와 대체로 일치. 그러나
**평가(Phase 3-4) 난이도를 구조적으로 오판**했다. 핵심 3가지:

1. LeWM planning 스택은 "goal 이미지와의 임베딩 MSE"가 cost의 전부인 **단기 goal-reaching
   전용** — 계획의 "centerline progress cost 정의"에 대응하는 인프라가 코드에 없음.
2. (a)안 전제인 렌더링이 vendored f110_gym에서 **전부 주석 처리**되어 있어 파이프라인이
   "구축 필요"가 아니라 "존재 0" — Phase 2까지 미루면 안 되고 Phase 1 스파이크 필요.
3. Python 3.8(기존 venv) vs 3.10(le-wm 요구) — 별도 venv는 "충돌 시"가 아니라 **확정**.
   swm 미설치 상태라 설치가 Phase 1 전체의 SPOF → step 0으로 당겨야 함.

---

## 1. 사실 검증 — 계획/021 서술 vs 실제 코드

### 일치 확인된 것 (계획 전제 유효)
| 주장 | 근거 | 판정 |
|---|---|---|
| 입력 = (RGB pixels, action), ViT cls token만 | jepa.py:34-38 (`last_hidden_state[:, 0]`) | ✅ |
| `action_encoder.input_dim = frameskip × action_dim` | train.py:68 | ✅ |
| num_steps = num_preds(1)+history_size(3) = 4, frameskip 5 | config/train/data/*.yaml, lewm.yaml `wm:` 블록 | ✅ |
| HDF5/.lance + `swm.data.load_dataset`, 이름→`$STABLEWM_HOME/<name>.h5` 해석 | train.py:56, README §Data | ✅ |
| encoder/predictor scratch 학습 (pretrained: false) | config/train/model/lewm.yaml | ✅ |
| CEM/Adam(GradientSolver) solver | config/eval/solver/{cem,adam}.yaml (둘 다 swm 클래스) | ✅ |
| reward 없이 학습 (loss에 reward 항 없음) | train.py:39-41 | ✅ |
| ViT-tiny(embed 192), patch 14, image 224, ~15M | model/lewm.yaml + README abstract | ✅ |

### 어긋난 것
- **EMA 서술 오류**: 계획 Phase 1 "loss, EMA" → LeWM에 EMA **없음**. 손실 = pred MSE +
  SIGReg(module.py:10-36, train.py:39-41). README abstract가 "EMA 없이 end-to-end 안정
  학습"을 핵심 기여로 명시. 021 §2엔 EMA 언급 없음 — 계획서 작성 중 혼입(DINO-WM/I-JEPA
  혼동 추정). (b)안 검토 시 "collapse 방지 장치 = SIGReg"라는 사실이 판단에 직결되므로 정정 필요.
- **"cost model"의 실체**: README:124-135의 `AutoCostModel`은 학습된 cost 모델이 아니라
  **JEPA 체크포인트 로더**. cost = `jepa.get_cost`(jepa.py:128-153) → `criterion`
  (jepa.py:112-126) = goal 임베딩과 마지막 step의 MSE. 그 이상도 이하도 아님.

---

## 2. Critical 발견 3건

### C-1. 평가 패러다임 불일치 (최대 리스크)
- eval.py:49-152: 평가 = `swm.World`에 **등록된** env(`swm/PushT-v1` 등)에서 데이터셋의
  한 시점을 `_set_state`로 복원 → `goal_offset_steps=25` step 뒤 프레임을 goal 이미지로 공급
  → eval_budget=50 내 도달 여부. **단기 goal-reaching이며 연속 lap 주행과 근본적으로 다름**.
- f1tenth는 swm env가 아님 → eval.py 재사용 불가. solver+get_cost는 재사용 가능하나
  env 루프·goal 공급·action 역정규화는 신규 구현.
- "progress cost"는 임베딩→진행도 매핑이 필요한 **신규 시스템**(코드에 없음).
- **수정**: 결정③을 분기 — (③-a) 레퍼런스 lap(best policy 에피소드)에서 N step 앞 프레임을
  subgoal로 공급하는 receding-horizon goal-chasing(정본 get_cost 유지, **권장**) vs
  (③-b) custom cost head 학습. 1차 milestone = 단기 goal-reaching 성공률(정본 평가 방식
  그대로), 2차 = 완주/lap time으로 계단화.

### C-2. 렌더 파이프라인 부재
- `gym/f110_gym/envs/f110_env.py:41-354`: pyglet import·render()·EnvRenderer **전부 주석**.
  `dreamer_f1tenth/envs/f1tenth_env.py:122`: `render_modes: []`. + WSL2 headless에서
  pyglet 1.5는 디스플레이 문제 추가.
- (a)안 실행 가능성은 코드 분석으로 판명 안 되고 프로토타입으로만 검증됨 → Phase 2 결정
  이후로 미루면 안 됨.
- **수정**: pyglet 복원 대신 **map occupancy + (x,y,θ) pose 직접 top-down 래스터 합성**
  (numpy/PIL, 결정적·headless-safe)을 1순위 후보로. Phase 1에 렌더 스파이크 추가
  (기존 eval npz pose 몇 개 → 224×224 생성 확인, 반나절 분량). 이 결과가 결정①의 진짜 판단 재료.

### C-3. venv 확정 분리 + swm 설치 = SPOF
- RL_project venv = **Python 3.8.10**(torch 2.4.1+cu124, 직접 확인). le-wm 요구 = **3.10**
  (`uv pip install stable-worldmodel[train,env]`, README:32-37). swm은 어디에도 미설치.
- Phase 1 핵심 산출물(swm 스키마 = 데이터셋 스펙 SSOT)이 설치 성공에 100% 의존.
  데이터 생성(py3.8) ↔ 학습(py3.10)에서 **.h5가 두 환경의 경계 인터페이스**가 됨을 명시해야.
- **수정**: Phase 1 step 0 = py3.10 venv + swm 설치 + **공식 pusht .h5 다운로드(HF,
  README:41-52) → train.py smoke 1회(레퍼런스 런)**. 스키마를 swm 소스 분석만이 아니라
  실물 공식 h5에서 h5dump로 역추출 → 스펙 신뢰도 격상 + 가장 싼 디리스킹.

---

## 3. Major 발견

- **M-1 (EMA)**: §1 "어긋난 것" 참조. Phase 1 항목을 "loss = pred MSE + SIGReg, target은
  detach 없는 동일 encoder"로 정정.
- **M-2 정규화/경계 contract 누락**: train.py:59-66 — pixels는 ImageNet 정규화(utils.py:6-10),
  **action 포함 모든 non-pixels 키 z-score**. train.py:25 `nan_to_num(action)` — 시퀀스 경계
  **NaN padding 규약** 전제. eval.py:71-82 — 평가 시 StandardScaler 재-fit. 데이터셋 스펙에
  (i) 경계 NaN 규약, (ii) planning 시 action 역정규화 항목 추가 필요(C-1 custom 루프에서
  역정규화 누락 = 전형적 silent failure).
- **M-3 시간 해상도 설계 결정 누락**: f1tenth env step=0.02s(action_repeat=2,
  f1tenth_env.py:84). frameskip=5 유지 시 model step=0.1s, CEM horizon 5 ×
  action_block 5(config/eval/pusht.yaml) → **lookahead 0.5초**. lap 6~13s racing에 빠듯.
  frameskip/horizon/receding_horizon/subgoal 간격은 한 묶음의 설계 결정 → Phase 2에
  결정④로 신설(Phase 3-2의 배선 항목이 아님).
- **M-4 (b)안 숨은 비용**: encode가 HF ViT API에 결합(jepa.py:37-38), 전처리 'pixels' 키
  고정(train.py:59), eval transform 'pixels'/'goal' 고정(eval.py:61-64), goal 인코딩도
  lidar화 필요. (b) = encoder 교체가 아니라 train/eval/swm 파이프 연쇄 수정. 결정① 비교표에
  명기. C-2 스파이크가 성공하면 (a)가 사실상 우세해지는 구조 → 스파이크 먼저, 결정은 그 후.

## 4. Minor 발견

- m-1: train/val 분리는 코드가 이미 함(train.py:73-79, 0.9 random_split) — 단 **window
  단위**라 같은 에피소드 인접 window가 양쪽에 들어가 val loss 낙관적. 게이트로 쓸 때 인지.
- m-2: `rollout(history_size=3)` 기본 인자 하드코딩(jepa.py:61) — history_size 변경 시
  planning 쪽 정합 주의.
- m-3: (a)안에서도 policy는 lidar+state 소비 → 수집 = "vector obs로 policy 구동 + 이미지
  병행 기록" 이중 구조. Phase 3-1 스펙에 명시.
- m-4: 용량 추정 부재 — 224×224×3 uint8, 300ep×400step ≈ 17GB급. dtype/압축은 공식
  pusht h5 모방(step 0이 해결).
- m-5: snapshot 전제 **이미 충족**(직접 확인): stage1_map_easy3 diversity bin 9개
  (6.1~18.4s) + best 6.1s, stage2_oschersleben 16.6s 존재. 형식적 재확인만.
- m-6: data yaml `${eval:'...'}` resolver는 OmegaConf 기본 아님 — spt/swm import가 등록
  추정. step 0 smoke에서 자연 검증.

## 5. 리스크 Top 3 + 완화

1. **MPC lap 주행 실패** (C-1) → ③-a subgoal-chasing으로 정본 패러다임 유지 +
   goal-reaching 성공률을 1차 milestone로 계단화.
2. **관측 모달리티/렌더** (C-2) → pose+map 래스터 스파이크를 Phase 1로 당김.
3. **swm 블랙박스** (C-3) → step 0 설치 + 공식 h5 역추출 + 레퍼런스 런 1회.

## 6. 과잉/과소

- **과소**: Phase 3-4(평가)가 한 줄인데 실제 최대 공수. "공식 데이터로 le-wm 레퍼런스 런"이
  계획에 없음(가장 싼 디리스킹).
- **과잉**: 결정② policy 믹스는 v1에서 정교할 필요 없음 — map_easy3 단일 트랙,
  best + 중간 bin 2~3개로 시작 후 반복 조정. 트랙 2종은 v2.

## 7. 계획 수정안 (diff 요약)

```diff
 Phase 1 — LeWM 코드 분석
+  0.  [신규, 최우선] py3.10 venv + stable-worldmodel[train,env] 설치
+      → 공식 pusht .h5 다운로드 → train.py smoke(레퍼런스 런) → 실물 h5 스키마 역추출
+  0b. [신규] 렌더 스파이크: map+pose → 224×224 top-down 래스터 프로토타입(pyglet 아님)
-  - jepa.py, module.py: ... loss, EMA
+  - jepa.py, module.py: ... loss(pred MSE + SIGReg; EMA 없음)
+  - 정규화 contract: action 포함 z-score, pixels ImageNet, 경계 NaN, eval scaler 재-fit

 Phase 2 — 설계 결정
-  결정 ①: Phase 1 결과를 보고 결정
+  결정 ①: 렌더 스파이크 결과로 결정 ((b)안에 "eval/planning 스택 연쇄 수정" 비용 명기)
+  결정 ② (v1 축소: map_easy3 단일, best+중간 bin 2~3개)
-  결정 ③: f1tenth용 cost 정의(centerline progress 등)
+  결정 ③: (③-a) 레퍼런스 lap subgoal-chasing(정본 get_cost 유지, 권장) vs
+          (③-b) custom cost head 학습. milestone 계단화(goal-reaching → 완주).
+  결정 ④ [신규] 시간 해상도: frameskip × horizon × receding_horizon × subgoal 간격
+          (dt=0.02s 기준 lookahead 초 단위 검토)

 Phase 3 — 구현
   1. (policy lidar 소비 + 이미지 병행 기록 이중 구조 명시; dtype/압축 = 공식 h5 모방)
-  4. 평가: f1tenth sim에서 MPC 주행 → 지표 산출
+  4. 평가 [공수 재산정]: f1tenth용 planning 루프 신규 구현
+     (eval.py 재사용 불가; solver+get_cost 재사용, env 루프·subgoal 공급·역정규화 신규)

-  전제 조건: Stage1/Stage2 snapshot 점검
+  전제 조건: 충족 확인됨(stage1 bin 9개+best 6.1s, stage2 16.6s) — 형식 재확인만
```

## 8. 확인 못 한 항목 (미검증 주장)

- **swm 내부 코드** (미설치): load_dataset 윈도잉/frameskip 소비 방식, h5 정확한 키·shape
  (`pixels` 저장 형식; `ep_idx`/`episode_idx`·`step_idx`는 eval.py:30에서 두 이름 허용만 확인),
  NaN padding 규약, WorldModelPolicy/CEMSolver의 정규화 처리, swm.World API → step 0이 해소 장치.
- **LeWM 논문 본문**(arXiv 2603.19312) 미조회 — frameskip 의미, 데이터 규모 권장치.
- uv/py3.10 설치 가능 여부, GPU VRAM 여유(WSL2 + cu124 torch py3.8 venv 동작 중이므로 GPU 가용).
- `f1tenth_LeWM/_thinking/env_setting` 문서 3개 미열람(검토 대상이 제시된 계획 텍스트였음).
- configs.yaml f1tenth 블록 action_repeat=2의 정밀 라인 추적 생략 — wrapper 주석
  (50 env step/s, dt 0.02s)으로 교차 확인.

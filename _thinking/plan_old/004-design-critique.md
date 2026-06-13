# 004 — Phase 2 설계서(plan/003) 비판적 검토 (critic 리뷰)

> 2026-06-13. plan/003-design-decisions.md(설계 결정 4개)에 대한 읽기 전용 검토 기록.
> 검증 대상 1차 소스: `~/le-wm` 정본(train.py/jepa.py/eval.py/config), swm 패키지 소스
> (`f1tenth_LeWM/.venv/.../stable_worldmodel/`), 공식 tworoom.h5 실측(h5py),
> 논문 v3 PDF(텍스트 스트림 추출), f1tenth env(`dreamer_f1tenth/envs/f1tenth_env.py`,
> vendored `gym/f110_gym`), snapshot 디렉토리, 렌더 스파이크 산출물, map_easy3 점유맵 실측.
> append-only.

---

## 0. 종합 평결: **수정 후 진행**

설계서의 사실 인용은 표본 검증 결과 **전부 1차 소스와 일치**했고(§1), 결정 ①②④는 근거
사슬이 견고하다. 그러나 결정③(평가)에 **구조적 공백 1건**이 있다: 평가 루프는 f1tenth
시뮬레이터와 LeWM planning 스택(swm, py3.10 전용)이 **같은 프로세스**에 있어야 하는데,
시뮬레이터는 py3.8 venv에만 설치돼 있고 vendored f110_gym은 module-level에서
gym 0.18 + pyglet을 import한다. 설계서(와 env_setting/004)의 "두 venv, .h5가 유일한 경계"
원칙은 **수집에는 성립하지만 평가에는 성립하지 않으며**, 이 모순이 설계서 어디에도 다뤄지지
않았다. 이 1건(C-1)을 Phase 3-1과 병행하는 디리스킹 스파이크로 당기고, 평가 루프의 swm
인터페이스 명세(M-1)와 action 기록 규약(M-2)을 보강하면 Phase 3 진입 가능하다.

---

## 1. 사실 검증 — 설계서 인용 vs 1차 소스

### 일치 확인 (설계서 전제 유효)

| 설계서 주장 | 1차 소스 근거 | 판정 |
|---|---|---|
| 로더 윈도우는 에피소드 내부만, 길이<span(=20) 에피소드 통째 제외 | swm `data/dataset.py:48-55` (`span = num_steps×frameskip`, `length >= self.span`) | ✅ |
| frameskip = 로더 파라미터: non-action 키 `[::frameskip]` 서브샘플, action은 전부 로드 후 블록 평탄화 → **h5 재작성 없이 frameskip 변경 가능** | `data/formats/hdf5.py:121-122` + `dataset.py:69-71` (`reshape(num_steps, -1)`) | ✅ |
| NaN 규약: 에피소드 마지막 row action만 NaN | **tworoom.h5 실측**: 920,809 rows 중 NaN row 정확히 10,000개, 전부 `ep_offset+ep_len-1` 위치와 일치. 소비처 train.py:25 `nan_to_num`, eval.py:77 scaler fit 제외 | ✅ |
| 윈도우 수 검산 730,809 | 실측 재계산 일치 (min ep_len=31이라 제외 에피소드 0) | ✅ |
| 공식 HDF5Writer 무압축 + chunks=(1,*shape) | `hdf5.py:276-286` (create_dataset에 compression 인자 없음) | ✅ |
| 정규화 왕복 자동: process z-score → CEM은 z-score 공간 → `inverse_transform` 후 반환 | `policy.py:437`, eval.py:75-77 (StandardScaler, NaN row 제외 fit) | ✅ |
| CEM 샘플 unbounded (클램핑 없음) | `solver/cem.py:191-198` (`randn*var+mean` 그대로 평가) | ✅ |
| planning context 1프레임 (history_len 기본 1, eval config 미지정) | `policy.py:31` (`history_len: int = 1`), `le-wm/config/eval/pusht.yaml`(horizon 5/receding 5/action_block 5, history_len 없음) | ✅ |
| goal_offset 25 step, receding 5×block 5=25와 정합 | `config/eval/pusht.yaml` `goal_offset_steps: 25` | ✅ |
| cost = goal 임베딩과 마지막 step MSE뿐 | `jepa.py:112-126` criterion | ✅ |
| action_dim=2 → action_encoder input 10 자동 배선 | `train.py:68` | ✅ |
| 'pixels' 키 고정 전처리, eval transform 'pixels'/'goal' 고정 | `train.py:59`, `eval.py:62-63` | ✅ |
| frameskip=5 의미 = action block grouping | **논문 v3 PDF 원문 확인**: "We apply a frame-skip of 5, grouping consecutive actions between frames into a single action block" | ✅ |
| "pseudo-expert or exploratory, 커버리지" 요건 | 논문 v3 원문 확인: "they may be pseudo-expert or exploratory, as long as they sufficiently cover the…" | ✅ |
| 10 epoch 충분 | 논문 v3 원문 확인: "10 epochs are sufficient to reach the best performance" | ✅ |
| 15M 단일 GPU | 논문 v3 원문 확인: "With 15M parameters trainable on a single GPU" | ✅ |
| predictor small이 best (Tab.6) | PDF 텍스트에서 "small 96.0 ± 2.2" 패턴 확인 (tiny/base 수치는 추출 실패 — §7) | ✅(부분) |
| snapshot 보유: diversity bin + best 6.1s | `runs/stage1_map_easy3/` 실측: policy_lap{6.1,7.0,8.0,9.0,10.3,13.1,15.2,16.2,17.1,18.4}s + policy_best_lap6.1s — **distinct lap 10개**(6.1 포함). "bin 9개+best"는 6.1 제외 셈법이면 일치 | ✅(표기 주의) |
| env step 0.02s (sim 0.01 × action_repeat 2) | `f1tenth_env.py:42-43` (SIM_TIMESTEP=0.01, DEFAULT_ACTION_REPEAT=2) | ✅ |
| env wrapper가 step에서 steer/speed clip | `f1tenth_env.py:306-307` | ✅ |
| npz에 pose 없음 → raw env에서 병행 기록 필요 | state 5키 정의(`f1tenth_env.py:53-58`)에 pose 없음 확인. raw pose는 `env._raw_obs["poses_x"]` 등으로 접근 가능(`f1tenth_env.py:290,469`) + info에 `arclen_s`/`closest_idx` 제공(`:457-458`) | ✅ |
| 시작 pose 다양화가 reset API로 가능 | `f1tenth_env.py:277-278` — `reset(options={"pose": (x,y,θ)})` 지원 | ✅ |
| 렌더 스파이크 SPIKE_OK | `spikes/out/ego_s*-1.png` 6장 실재, 육안 확인(단 시인성 문제 실재 — P-1 답변 참조) | ✅ |

### 어긋남 / 불완전 (발견 사항으로 격상)

- 설계서 ③ "재사용: WorldModelPolicy… policy가 env에 요구하는 건 action_space/num_envs뿐"
  (analysis/002 §3 인용)은 **불완전** — get_action 경로는 vector-env 형태 인터페이스를
  추가로 요구한다 → M-1.
- 설계서·env_setting/004의 "**.h5가 두 venv의 유일한 경계**"는 수집에만 참 — 평가는 시뮬과
  planner가 한 프로세스여야 해서 경계가 무너진다 → C-1.
- plan/002 헤더의 "vendored f110_env.py pyglet 전부 주석" 서술은 부정확: 파일 상단 사본
  (27-30행)만 주석이고 **실제 모듈은 `f110_env.py:394-410`에서 gym 0.18 + pyglet을 활성
  import** 한다(py3.8 venv에 pyglet 1.5.0 설치 확인). 설계서 자체 주장은 아니지만 C-1의
  근거 사슬에 걸리므로 기록.

---

## 2. 설계 결정 4개 평가 (요약)

- **결정① (ego-centric top-down 래스터)**: 타당, 확정 동의. 정본 무수정 목표에 유일하게
  부합하고 스파이크로 검증됨. 숨은 전제 2개를 명시할 것 — (i) 단일 프레임 planning context에서
  속도가 이미지에 비관측(M-3), (ii) goal-conditioned cost가 이미지의 **장소 식별력**에
  의존(P-1). npz→재렌더 분리 설계 덕에 렌더 스펙 결정들이 전부 저비용 가역이라는 점이 이
  결정의 가장 큰 강점이고, 그 강점을 게이트에 활용하는 권고를 P-1에 담음.
- **결정② (map_easy3 1M, 3-소스 믹스)**: 규모·트랙 선택 타당(paper 최소 사례와 동급,
  TwoRoom=920k 실측 확인). 믹스 구성도 방향은 맞으나 ε 설계와 게이트 지표가 비어 있음(P-2),
  action **기록 스케일 규약**이 미명시(M-2), 시작 pose 섭동의 충돌 안전 규칙 부재(m-4).
- **결정③ (receding subgoal-chasing + M0/M1/M2)**: 패러다임 선택(정본 cost 유지, 계단식
  milestone)은 옳다. 그러나 실행 계획에 구멍 — 평가 런타임(C-1), swm 인터페이스 상세(M-1),
  goal 공급의 단일 lap 의존(P-5 → 합성 goal 권고), M1 판정 보강(P-3). M0 decoder는 약간
  과잉 — 저비용 대안 선행 권고(m-6).
- **결정④ (frameskip 5 유지)**: 타당, 확정 동의. "데이터 재사용 가능" 주장 소스 레벨 확인.
  단 frameskip 10 전환 시 부수 효과 명시 필요: span 20→40(짧은 에피소드 추가 탈락),
  action_encoder input 10→20(**모델 재학습**), PlanConfig.action_block도 10으로 동행 변경.
  또한 "lookahead 0.5s가 짧다"보다 먼저 의심할 것은 M-3(속도 비관측)이다 — 실패 패턴 진단
  순서에 반영할 것.

---

## 3. 발견 사항

### Critical

**C-1. 평가(M1/M2) 런타임 미설계 — py3.10 프로세스에 f1tenth 시뮬레이터가 없다**
- [근거] 평가 루프는 `env.step ↔ policy.get_action`이 한 프로세스에서 교차해야 함(설계서 ③
  구조 자체). swm은 py3.10 전용 문법 사용(`policy.py:106` `np.ndarray | np.generic` 등)이라
  py3.8에서 구동 불가. 반대로 f110_gym은 py3.10 venv에 미설치(import 실측: gym/numba/f110_gym
  전부 ModuleNotFoundError)이고, vendored `gym/f110_gym/envs/f110_env.py:394-410`이
  module-level에서 `import gym`(setup.py가 gym==0.18.0 고정) + `import pyglet`을 수행.
  gym 0.18은 py3.10 + 최신 setuptools에서 설치가 깨지는 세대의 패키지라 단순 pip로 안 됨.
- [문제] Phase 3 작업 분해에서 3-4에 도달해서야 발견될 SPOF. 설계서·env_setting/004의
  "두 venv, .h5 경계" 원칙이 평가 단계에서 무너지는데 어디에도 언급 없음.
- [수정 제안] **Phase 3-0 스파이크 신설**(3-1a와 병행, 반나절): py3.10 venv에
  `pip install gym==0.18.0 --no-deps` + `pyglet==1.5.x` + `numba` + f110_gym editable 설치
  → wrapper import → 1 에피소드 무policy rollout smoke. 게이트 = obs/step 결과가 py3.8과
  일치(같은 seed/pose에서 lidar/state 비트 단위 비교). 실패 시 fallback 확정:
  (i) py3.8 env 서버 ↔ py3.10 planner의 소켓/파이프 브리지(주고받는 것은 pose·obs·action
  뿐이라 인터페이스 작음), 또는 (ii) f110_gym을 LeWM 프로젝트에 포팅(gym.Env→gymnasium.Env,
  pyglet import 제거 — 물리 코드 무변경이므로 사용자 제약 위반 아님).

### Major

**M-1. WorldModelPolicy의 env 인터페이스 가정 — "action_space/num_envs뿐"이 아니다**
- [근거] `policy.py:362` `n_envs = self.env.num_envs`(getattr 폴백 **없음** — set_env의
  `policy.py:337`과 다름), `policy.py:427` `self.env.single_action_space.shape[-1]`,
  `policy.py:434` `action.reshape(*self.env.action_space.shape)`,
  `solver/cem.py:64` `action_dim = prod(action_space.shape[1:])` — **배치형(vector env)
  action_space 전제**. 단일 raw env의 Box(2,)를 주면 cem.py:64에서 action_dim=1로 잘못
  계산되어 silent shape 오류.
- [문제] 설계서 ③의 신규 구현 명세("env reset → infos 구성 → get_action")만으로는 첫 실행에서
  AttributeError/shape 불일치. analysis/002 §3의 낙관적 요약을 그대로 인용한 결과.
- [수정 제안] 평가 루프 명세에 **VecEnv 셔임** 1줄 추가: `num_envs=1`,
  `single_action_space=Box(low,high,(2,))`, `action_space=Box(low,high,(1,2))`를 가진 thin
  wrapper. infos의 모든 값은 leading env dim 포함 (pixels `(1,1,224,224,3)` uint8 HWC,
  proprio `(1,1,5)`, goal `(1,1,224,224,3)`).

**M-2. action 기록 규약 미명시 — 어떤 스케일·어느 시점의 action을 쓰는가**
- [근거] snapshot policy는 정규화 obs를 소비하고 [-1,1]계 action을 내며, wrapper가 raw
  물리 스케일로 매핑 후 `f1tenth_env.py:306-307`에서 clip. 한편 평가 시
  WorldModelPolicy는 `process['action']`(z-score)을 **데이터셋의 action 분포**로 fit해
  역변환 후 env에 반환(`policy.py:437`).
- [문제] 데이터셋에 정규화 스케일 action을 기록하면 평가 시 역정규화 결과가 물리 스케일과
  어긋나는 전형적 silent failure. 또 noise 믹스(소스 3)에서 "정책 출력"과 "실행된 action"이
  달라짐.
- [수정 제안] 규약 명시: **h5의 action = env에 실제 적용된 raw 물리 스케일 (steer[rad],
  speed[m/s]), noise·clip 반영 후 값**. 3-1a 게이트에 "기록 action을 그대로 재주입하면 동일
  궤적 재현(결정적 sim)" 검증 1회 추가 — pose 무결성과 action 무결성을 동시에 잡는 가장 싼
  테스트.

**M-3. planning context 1프레임 + ego-centric heading-up 렌더 = 초기 속도 비관측**
- [근거] planning 시 모델 입력은 현재 프레임 1장(`policy.py:31` history_len=1). ego-centric
  heading-up 이미지 1장에는 속도·요레이트 정보가 0이다. 학습 시에는 context 3프레임이라
  문제없지만, 추론 초기 상태에는 속도가 없다. 더구나 rollout 구조상 history_len>1로 올리는
  것은 해결책이 아님 — `jepa.py:72` `act_0, act_future = split(action_sequence, [H, T-H])`가
  **후보(계획) action의 앞 H블록을 과거 action으로 소비**하므로, H>1이면 실제 과거가 아니라
  CEM이 샘플한 가짜 과거가 들어간다. config 변경만으로 정합하게 풀 수 없는 구조.
- [문제] 완화 요인은 있다: f1tenth action의 speed는 **절대 목표 속도**(PID 추종)라 행동
  시퀀스가 미래 속도를 대부분 결정한다. 그러나 가감속 한계·PID 과도구간(고속→저속 코너 진입
  등) 동안은 초기 속도를 모르면 위치 예측이 수 m 어긋날 수 있음. 8m/s에서 0.5s면 4m다.
- [수정 제안] (a) v1은 그대로 가되 **M0 진단 항목에 명시**: open-loop rollout을 "같은 위치,
  다른 진입 속도" 컨텍스트로 비교해 모델이 속도를 어떻게 추정하는지 확인. (b) M1 실패 시
  1순위 처방으로 **렌더 속도 글리프**(차량 마커 앞 속도 비례 선분, 요레이트는 곡률로) 예약 —
  렌더 스펙 추가일 뿐이라 정본 무수정 원칙 유지, 단 학습 데이터부터 다시 렌더(npz 재렌더로
  수집 재실행은 불필요).

**M-4. goal 공급원을 "레퍼런스 lap 프레임"에서 "centerline s-indexed 합성 렌더"로 바꿀 수 있다 (P-5와 동일 결론)**
- [근거] 관측 자체가 합성 렌더이므로, goal 이미지도 **임의 pose에서 렌더러로 직접 생성
  가능** — 학습 분포와 정확히 같은 렌더러를 쓰므로 분포 이탈 없음. centerline csv(s,x,y,tx,ty)
  가 pose 합성 재료를 이미 제공(스파이크가 실증).
- [문제] 레퍼런스 lap 1개 의존은 (i) 단일 주행선 고정, (ii) 이탈 시 time-index 동기화 깨짐,
  (iii) best lap의 공격적 라인(벽 근접)을 추종 목표로 강제하는 3중 약점.
- [수정 제안] M1은 정본 모사 목적이므로 레퍼런스 lap 방식 유지하되, **M2(완주 체인)는
  s-indexed 합성 goal**로: goal = render(centerline pose at `s_now + Δs`), Δs ≈ 2.5~4m,
  `s_now`는 env info의 `arclen_s`(`f1tenth_env.py:457`) 사용. lap당 어떤 lap 녹화도 불필요해지고
  과속/저속 정책 차이에도 일정한 "당근" 거리가 유지된다.

### Minor

**m-1. infos에 'action' 키 필수 — 없으면 get_cost가 KeyError로 즉사**
- [근거] `jepa.py:145` `goal.pop("action")` — 기본값 없는 pop. CEM의 expanded_infos는 평가
  루프가 준 infos에서 옴. (값은 rollout에서 `jepa.py:73` `info["action"] = act_0`로 덮이므로
  **존재만 필요, 내용 무관**.)
- [수정 제안] 평가 루프 infos 명세에 `action: (1,1,2) f32` (직전 실행 action 또는 0) 추가.
  process에 'action' scaler가 있으므로 z-score 통과 후 버려진다 — 무해.

**m-2. 에피소드 경계/subgoal 전환 시 action buffer 플러시 규약 미명시**
- [근거] `policy.py:364-370` — `_needs_flush` 키로 buffer clear. subgoal 갱신 주기 25 step이
  receding 25 step과 정렬돼 평시에는 자연 소진되지만(설계서가 정확히 인지), 에피소드 reset·
  M1 trial 반복 시에는 flush 또는 policy 재생성이 필요.
- [수정 제안] 평가 루프 명세에 "reset마다 `infos['_needs_flush']=[True]` 주입" 1줄 추가.

**m-3. M1 판정에 유클리드 거리 단독은 자기교차 트랙에서 오판 가능**
- [근거] map_easy3는 자기교차 트랙(wrapper의 windowed closest-point 로직 자체가 그 증거,
  `f1tenth_env.py:96-99` 주석). 교차 구간에서 다른 가지 위의 차가 goal pose와 1.0m 이내일 수 있음.
- [수정 제안] 판정 = pose 거리 ≤ 1.0m **AND** |Δs| ≤ 2.0m (arclen_s 진단 컬럼 활용). P-3 참조.

**m-4. 시작 pose 횡방향 섭동의 충돌 안전 규칙 부재**
- [근거] centerline clearance 실측(점유맵 distance transform): 중앙값 1.37m, p5 0.94m,
  p1 0.90m, 최소 0.22m(단 <0.5m는 s≈100.26~100.52m 구간 12점뿐 — start/finish 부근 국소
  아티팩트로 추정). 무제약 횡 섭동은 벽 안 스폰 가능.
- [수정 제안] 횡 오프셋 한도 = min(0.6m, clearance(s)−0.45m) (차폭 ~0.3m + 여유), heading
  섭동 ±15°. clearance(s)는 본 검토에서 쓴 distance transform으로 1회 사전 계산해 csv에 붙이면
  됨. + reset 직후 `collision_raw` 확인해 충돌 스폰은 재추첨 (`ignore_first_collision`이 이를
  가려버리는 점 주의 — `f1tenth_env.py:321-323`).

**m-5. M1 베이스라인에 "open-loop 레퍼런스 재생" 상한 추가**
- [문제] random policy는 하한일 뿐 하니스 자체 검증을 못 한다.
- [수정 제안] 같은 시작 pose에서 레퍼런스 lap의 기록 action을 그대로 재생(결정적 sim이므로
  100% 도달이 정상). 이게 깨지면 모델이 아니라 평가 루프/렌더/정규화 버그 — 디버깅 순서를
  앞당겨주는 가장 싼 장치이며 M-2의 action 무결성 게이트와 겸용 가능.

**m-6. M0 decoder는 절반만 필요 — retrieval 진단을 선행**
- [문제] 사후 decoder 학습은 별도 학습 루프 구현 비용이 있음(개인 프로젝트 규모에서 1~2일).
- [수정 제안] M0-lite 선행: 데이터셋 프레임 임베딩 인덱스(부분 샘플 ~10k장)를 만들어 open-loop
  rollout의 예측 임베딩마다 **nearest-neighbor 실프레임 retrieval**로 시각화. 학습 0, 반나절.
  트랙 구조 상상 여부 + place aliasing 여부(P-1)를 동시에 진단. decoder는 M0-lite가 애매할
  때만.

**m-7. 표기 정리**
- snapshot bin 개수: 파일 기준 distinct lap 10개(6.1 포함). "bin 9개(6.1~18.4)+best(6.1)"
  표기는 6.1 중복 셈/제외 셈이 혼재 — 믹스 비율 계산 시 분모가 달라지므로 한 줄로 확정할 것.
- M1 판정에 budget 미정의 — 정본 모사라면 `eval_budget=50 env step`(goal_offset의 2배,
  config/eval/pusht.yaml 패턴)을 명시.
- eval용 yaml에 `dataset.keys_to_cache=[action, proprio]`도 우리 데이터셋 기준으로 명시
  필요(eval.py:71-82가 이 목록으로 scaler를 fit — proprio (N,5)는 2D라 NaN 필터
  `eval.py:77` 통과에 문제없음).

---

## 4. 설계서의 검토 요청 포인트 5개 — 직접 답변

**P-1. 렌더 시야 폭 22.4m vs 12m**
**권고: 22.4m 유지 + 시인성을 "최종 224px 기준"으로 정량 보장.** 이유: goal-conditioned
cost는 이미지의 장소 식별력이 생명인데, 좁은 ego-view 복도 이미지는 직선 구간에서 자기유사
(place aliasing) 위험이 크다 — 22.4m면 맵(30×33m)의 ~절반이 보여 위치 시그니처가 강하다.
시인성은 FOV가 아니라 벽 두께의 문제다: 스파이크 이미지(`ego_s42m-1.png`) 육안 확인 결과
벽이 hairline(다운샘플 후 ~1px)이라, "원본 3px dilation"은 22.4m FOV의 5× 다운샘플에서
~1.5px로 부족하다. **스펙을 "최종 이미지에서 벽 두께 ≥2px"로 정의**하고 원본 해상도 dilation을
역산(22.4m이면 총 두께 ~10px=0.2m)할 것. 렌더 게이트에 육안 검증 외에 "벽 픽셀 비율/두께
자동 검사"를 추가. 이 결정은 npz 재렌더 설계 덕에 **저비용 가역**이므로 v1은 22.4m로 가고,
M0-lite retrieval에서 aliasing/시인성 어느 쪽이 문제인지 보고 바꾸면 된다(12m 축소는
aliasing을 키우는 방향이므로 마지막 수단).

**P-2. 믹스 비율 60/20/20과 noise ε**
**권고: 비율은 그대로(근거 수준에서 충분), 설계 에너지는 ε와 게이트 지표에 쓸 것.**
비율 자체는 paper 요건("커버리지")에 1차 근사로 충분하고 민감하지 않다. 구체화 제안:
(i) ε는 고정값 대신 **에피소드별 샘플링** — σ_steer ~ U(0.03, 0.12) rad, σ_speed ~ U(0.5,
2.0) m/s. 고정 ε는 한 가지 이탈 모드만 만든다. (ii) 파일럿(50ep) 게이트 지표를 명시:
noise 소스의 충돌 종료율 10~25% 목표(너무 낮으면 벽 근접 상태 미커버, 너무 높으면 20-step
미만 폐기 손실 증가), 횡방향 오프셋·속도 히스토그램이 bin 정책들 사이 공백을 메우는지 확인.
(iii) 비율보다 효과가 큰 것은 **시작 s-위치의 균등 스트래티파이**(소스·bin별로 s를 균등
분할)다 — 이미 설계에 있는 "랜덤 s"를 "균등 격자 + 지터"로 한 단어만 강화할 것.

**P-3. M1 판정 거리 1.0m**
**권고: 1.0m 채택 + |Δs| ≤ 2.0m 가드 + budget 50 env step 명시.** 실측 근거: centerline
clearance 중앙값 1.37m(≒트랙 반폭), p5 0.94m. 즉 1.0m는 "트랙 반폭 수준의 도달 판정"으로
적정하고, 더 좁히면(0.5m) 초기 모델에 과도, 더 넓히면(1.5m) 트랙 폭을 초과해 무의미.
단 유클리드 단독은 자기교차 구간에서 다른 가지를 "도달"로 오판할 수 있으므로(m-3) arclen_s
가드를 함께. 판정 시점은 "budget 내 최초 충족"(정본 terminated 패턴 모사).

**P-4. 에피소드 400 step 고정 vs 가변**
**권고: 가변(terminated 조기 종료 허용) + cap 400 + <20 step 폐기.** 사실 "고정"은 선택지가
아니다 — 충돌 시 sim이 종료되므로(충돌 후 데이터는 존재하지 않음) 고정 길이를 강제하면
충돌 에피소드를 전부 버리거나 충돌 직전 상태를 체계적으로 과소표집하게 된다. noise 소스의
존재 이유가 바로 벽 근접·회복 상태 커버리지이므로 조기 종료 에피소드는 **버리지 말고 포함**
해야 한다(20 step 이상이면 로더가 정상 소화 — tworoom도 min 31 step 가변 길이로 동일 패턴).
분포 영향은 게이트로: 파일럿 리포트에 ep_len 히스토그램 + 폐기율(<20 step) 추가. 폐기율이
높으면(>10%) noise ε 상한을 낮추는 식으로 ε와 연동 조정.

**P-5. 레퍼런스 lap 1개 의존**
**권고: 위험 실재 — M2의 goal 공급을 s-indexed 합성 렌더로 교체(M-4).** 단일 lap 의존의
실질 위험은 "주행선 고정"보다 **시간 인덱스 동기화**다: 차가 레퍼런스보다 느리거나 이탈하면
"+25 step 앞 프레임"은 도달 불가능하게 멀어지고, receding 재계획마다 격차가 누적된다.
s-indexed(현재 arclen_s + Δs) 합성 goal은 이 문제를 구조적으로 제거하고, 렌더러가 곧
goal 생성기이므로 추가 비용이 0이다. M1은 "정본 평가 방식 모사"가 목적이므로 레퍼런스 lap
방식을 유지하되, 그 경우에도 best(공격적 라인) 대신 **중간 bin(8~9s) lap**을 레퍼런스로
쓰는 편이 추종 여유가 있다. best lap은 비교용 상한으로만.

---

## 5. 리스크 Top 3 + 설계서에 없는 완화책

1. **평가 런타임 부재 (C-1)** — Phase 3-4에서 발견되면 일정 전체가 막힘.
   완화: Phase 3-0 import 스파이크로 당기기 + 실패 시 브리지/포팅 fallback 사전 확정 (C-1 참조).
2. **goal cost의 유효성 (P-1 aliasing × M-3 속도 비관측 × P-5 동기화)** — M1 실패가 "모델이
   나쁨"이 아니라 평가 설계 결함일 수 있는 3중 교란.
   완화: M0-lite retrieval 진단(m-6)으로 aliasing을 학습 직후 분리 진단 + open-loop 재생
   상한 베이스라인(m-5)으로 하니스 결함을 모델 결함과 분리 + s-indexed 합성 goal(M-4).
   M1 실패 시 진단 순서를 문서에 박을 것: 하니스(상한 베이스라인) → aliasing(M0-lite) →
   속도(M-3 진단) → 시간해상도(결정④ v2).
3. **offline OOD: CEM unbounded 샘플 + 벽 관통 상상** — 모델은 벽 통과 데이터를 본 적이
   없으므로 관통 계획의 비용 평가가 무의미(환각)할 수 있고, unbounded 샘플(cem.py:191)은
   z-score 역변환 후 물리 범위 밖 action을 만든다(env clip이 실행은 막지만 **cost 평가는
   OOD action으로 이미 이뤄짐**).
   완화: 설계서의 "필요시 solver 클램핑"을 **M1부터 기본 적용**으로 격상 — CEM 샘플을
   z-score 공간 ±2σ로 clip하는 콜백(스택 외부 코드, 정본 무수정). noise 데이터(소스 3)가
   벽 근접 상태의 실데이터를 공급하는 것이 구조적 완화임을 명시.

---

## 6. 과잉/과소 (개인 프로젝트 규모 대비)

- **과소**: 평가 런타임 계획(C-1), 평가 루프 인터페이스 상세(M-1, m-1, m-2), action 기록
  규약(M-2). 공통점 — 결정③ 주변의 "구현하면 알게 될 것들"이 사실은 설계 항목이었다.
- **과잉 (소폭)**: (i) 본 수집 2,500 ep을 한 번에 — 수집·렌더가 재실행 가능한 구조이므로
  **1차 500 ep(~200k transition)로 학습→M0-lite까지 한 바퀴** 돌고 렌더 스펙을 확정한 뒤
  잔여 2,000 ep을 채우는 2단 수집이 디스크·시간 리스크를 줄인다(파일럿 50 ep과 본 수집
  사이의 중간 게이트). (ii) M0 decoder 학습은 M0-lite로 대체 가능(m-6). (iii) Oschersleben
  v2 보류, frameskip 실험 v2 보류 판단은 둘 다 적절.

---

## 7. 설계서 수정안 (diff 요약)

```diff
 결정① 렌더 스펙 v1
-  벽 라인 dilation(원본 해상도에서 3px) 후 다운샘플
+  벽 두께 스펙 = 최종 224px 기준 ≥2px (22.4m FOV면 원본 ~10px dilation 역산)
+  게이트에 벽 픽셀 두께/비율 자동 검사 추가 (육안 검증 보완)
+  [숨은 전제 기록] 단일 프레임 planning context에서 속도 비관측 — M0 진단 항목,
+  실패 시 1순위 처방 = 속도 글리프 (npz 재렌더로 대응)

 결정② 데이터셋
+  action 기록 규약: env에 실제 적용된 raw 물리 스케일(noise·clip 반영 후) — 게이트에
+  "기록 action 재주입 → 동일 궤적 재현" 검증 추가
-  best policy + action noise (ε-Gaussian)
+  best policy + action noise (에피소드별 σ_steer~U(0.03,0.12)rad, σ_speed~U(0.5,2.0)m/s;
+  파일럿 게이트: noise 소스 충돌종료율 10~25%, ep_len 히스토그램·폐기율 리포트)
-  시작 pose: centerline 랜덤 s + 소량 횡/방향 섭동
+  시작 pose: 균등 s-격자+지터, 횡 오프셋 ≤ min(0.6m, clearance(s)−0.45m), heading ±15°,
+  스폰 충돌 시 재추첨
+  에피소드 길이: 가변(terminated 조기 종료 포함, cap 400, <20 step 폐기)으로 확정
+  본 수집 2단화: 500 ep → 학습+M0-lite → 스펙 확정 → 잔여 2,000 ep

 결정③ 평가
+  [신규] 평가 런타임: Phase 3-0 스파이크 — py3.10에 f110_gym 구동 (실패 시 브리지/포팅)
+  [신규] VecEnv 셔임 명세: num_envs=1, single_action_space, 배치형 action_space;
+  infos = {pixels(1,1,224,224,3)u8, proprio(1,1,5), goal(1,1,224,224,3), action(1,1,2),
+  reset 시 _needs_flush}
-  M1 판정 = pose 거리 1.0m
+  M1 판정 = pose 거리 ≤1.0m AND |Δs| ≤2.0m, budget 50 env step 내 최초 충족
+  M1 베이스라인 = random(하한) + 레퍼런스 action open-loop 재생(상한, 하니스 검증 겸용)
+  M1 레퍼런스 lap = 중간 bin(8~9s) (best는 비교용)
-  M2: 레퍼런스 lap subgoal 체인
+  M2: s-indexed 합성 goal — render(centerline pose at arclen_s+Δs), Δs≈2.5~4m
-  M0: 사후 decoder open-loop 시각화
+  M0-lite(선행): 예측 임베딩 → 데이터셋 NN-retrieval 시각화 (decoder는 애매할 때만)
-  CEM unbounded → env clip 의존, 필요시 클램핑
+  CEM 클램핑(z-score ±2σ, 외부 콜백)을 M1부터 기본 적용 (OOD cost 평가 차단)
+  M1 실패 진단 순서 고정: 하니스 → aliasing → 속도 → 시간해상도

 결정④
+  frameskip 10 전환 시 부수효과 명시: span 40 (짧은 에피소드 탈락↑), action_encoder
+  input 20 (재학습), PlanConfig.action_block=10 동행 변경

 Phase 3 작업 분해
+  3-0: py3.10 f110_gym 구동 스파이크 (3-1a와 병행) | 게이트: 1 ep rollout + py3.8 결과 일치
```

---

## 8. 확인 못 한 항목 (침묵 금지)

- **논문 Fig.15(embed dim 임계 ~184)·Tab.6 tiny/base 수치**: PDF 텍스트 스트림 추출로
  "small 96.0±2.2"와 15M·10 epochs·frameskip·pseudo-expert 문구는 원문 확인했으나, 그림
  기반 주장(Fig.15)과 표의 나머지 셀(80.67/86.7)은 추출 실패. analysis/001의 기록을 신뢰하되
  재검증은 못 함 — 단 이 수치들은 "정본 유지" 결정의 근거라 틀려도 결정 방향이 바뀌지 않음.
- **gym 0.18 + pyglet 1.5의 py3.10 설치 가능 여부**: 읽기 전용 검토라 설치 시도 안 함.
  C-1 스파이크가 답할 항목.
- **py3.8 rollout 스크립트(snapshot 로드 코드)의 실재**: `runs/stage1_map_easy3/eval_eps`
  존재로 간접 추정만. dreamer policy 추론 코드 재사용 가능성은 RL_project 쪽 코드 미열람.
- **blosc 압축률 ~14GB 추정**: 미실측. 라인아트 이미지는 tworoom(텍스처 장면 12GB/920k)보다
  잘 눌릴 것으로 추정되나 파일럿에서 실측할 항목 (설계서도 동일 인지).
- **렌더 1M장 처리 시간**: 미실측 (파일럿 항목).
- **clearance 실측의 좌표 변환 가정**: map origin lower-left, y-flip 관행을 가정하고 계산
  (centerline 4,475점 전부 in-bounds + clearance 분포가 물리적으로 그럴듯해 간접 교차검증됨).
  렌더 스파이크와 동일 변환인지 render_spike.py와의 대조는 생략.
- **Oschersleben 관련 전부** (v2 범위라 미검토).

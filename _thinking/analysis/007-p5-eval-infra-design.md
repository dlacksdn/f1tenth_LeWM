# 007 — P5 평가 인프라 설계 (plan_f1tenth.py): GuidedPolicy ↔ f110_gym MPC + normalizer 고정

> 2026-06-19. P5 understand workflow(opus 4축: plan_guided / GuidedPolicy / normalizer_fix / f110_eval)
> 결과 정리. **plan_f1tenth.py 구현용 1차 참조**(새 세션이 재실행 없이 바로 구현). 코어 무변경, 글루만.
> 사용자 규칙: 작성만, git add/commit/push는 지시 시. (handoff 아님 = P5 설계 산출물.)

---

## 0. 한 줄
Diffuser 모델로 f110_gym **2랩 주행 평가** = eval_gate.py `make_env` 체인 재사용 + 매 step
GuidedPolicy 호출(obs 133D min-pool 변환, **raw action→NormalizeActions 재정규화**) + normalizer 통계
pickle 고정. cap-5 baseline(2랩 107.16s)과 비교.

## 1. plan_f1tenth.py 골격 (A — plan_guided.py 23-69 재사용 + 76-118 교체)
- **로딩부 재사용**: `load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, epoch, seed)`
  → `DiffusionExperiment(dataset, renderer, model, diffusion, ema, trainer, epoch)`. diffusion=exp.ema,
  dataset=exp.dataset, renderer=exp.renderer(NullRenderer). (serialization.py:36-60)
- **value 사용 분기**: `USE_VALUE`면 value load_diffusion + check_compatibility + ValueGuide +
  GuidedPolicy(sample_fn=`sampling.n_step_guided_p_sample`, guide, scale, n_guide_steps, t_stopgrad).
  미사용이면 value_loadpath/pkl 없으면 FileNotFoundError → **분기로 value 로딩 자체 skip**.
- **★ unconditional 함정**: GuidedPolicy.__call__(policies.py:28)이 무조건 `diffusion_model(cond,
  guide=self.guide, **kwargs)` 호출 → default_sample_fn은 guide kwarg 미수용 TypeError. 우회 2택:
  (a) GuidedPolicy에 `guide is None` 분기 추가(깔끔, 코어 수정), **(b) value model 로드 + scale=0.0**
  (변경 0, guidance 무효화 — value 학습은 하니 (b) 권장). 즉 **P4에서 value도 학습 → value guidance 사용**.

## 2. GuidedPolicy 호출 규약 (B)
- 입력 `conditions = {0: obs133}` — **1D 벡터(batch 차원 금지)**, einops repeat가 1D 가정. obs133 =
  concat(min-pool lidar 128, state 5).
- 흐름: normalize(obs) → batch_size(=64) 복제 → 역확산 가이드 샘플 → `trajectories[:,:,:action_dim]`
  → `normalizer.unnormalize('actions')` → **action[0,0]만 실행(receding-horizon MPC)**. (policies.py:23-42)
- ★ **출력 action = raw [steer rad, speed m/s]** — P3에서 로더가 action을 raw로 역정규화 후 normalizer를
  fit했으므로 unnormalize 출력이 곧 raw 물리값(추가 변환 불필요).
- batch_size=64는 후보 궤적 수, value 기준 sort 후 **best 1개** 출력. → value 부실하면 MPC 왜곡(value 품질 중요).
- **device 파라미터화 2곳**: ① policies.py:55 `'cuda:0'` → `self.device`(이미 프로퍼티 존재) ②
  arrays.py:6-7 전역 `DEVICE='cuda:0'` → `os.environ.get('DIFFUSER_DEVICE','cuda:0')`. ①만 고쳐도 추론
  핫패스는 device 무관(but to_device/batchify 다른 경로 위해 ② 권장).
- ★ **latency**: horizon 128(50Hz=2.56s) + 20 diffusion steps × batch 64 × n_guide 2 → 실시간 50Hz(20ms)
  초과 가능. **실측 필요**(평가는 sim이라 wall-clock 무관하나 보고서에 기록).

## 3. normalizer 고정 (C — P3 review major #2, 코어 무변경)
문제: eval 시 load_diffusion이 dataset_config(seed)로 SequenceDataset 재생성 → DatasetNormalizer가
**현재 디스크 데이터 min/max로 재fit**(serialization.py:47, 통계 저장 없음). Trainer.save는 model/ema/step만.
데이터/downsample 변경 시 train↔eval min/max 어긋남 → action/obs 디코딩 왜곡.

**해결(글루 2지점, 코어 trainer/serialization 무변경):**
- **저장(학습 글루)**: train.py `dataset = dataset_config()` 직후 삽입 —
  ```python
  import pickle, os
  _np = os.path.join(args.savepath, 'normalizer.pkl')
  with open(_np, 'wb') as f: pickle.dump(dataset.normalizer, f)
  ```
  (train_values.py도 동일 — 같은 데이터면 같은 통계라 무해.) DatasetNormalizer는 numpy/scipy라 pickle 가능.
- **로드(eval 글루, plan_f1tenth.py)**: load_diffusion 후 —
  ```python
  _np = os.path.join(args.loadbase, args.dataset, args.diffusion_loadpath, 'normalizer.pkl')
  if os.path.exists(_np):
      with open(_np,'rb') as f: dataset.normalizer = pickle.load(f)   # 재fit→학습시점 통계로 교체
  else: print('[glue] normalizer.pkl 없음 → 재fit(legacy) — 불일치 위험')
  ```
  이후 GuidedPolicy(normalizer=dataset.normalizer). load_diffusion 직접 수정 회피(d4rl 경로 파일부재 리스크).

## 4. ★ f110_gym 평가 루프 (D — 핵심 난관)
eval_gate.py(`build_config`/`make_env`/`run_episode`/`aggregate_episodes`/`is_completed`) 재사용,
agent 호출부만 GuidedPolicy로 교체.
- **env**: `config = build_config('f1tenth_Oschersleben')`; `config.v_max = 5.0`(cap-5 baseline action space);
  `env = make_env(config,'eval',0)`(F1Tenth→NormalizeActions→TimeLimit→SelectAction→UUID).
- **★ obs 변환(매 step, 133D)**:
  ```python
  def env_obs_to_cond(obs):
      lidar = np.asarray(obs['lidar'], np.float32).reshape(1,-1)   # (1,1080) 이미 [0,1]
      lidar128 = _downsample_lidar(lidar, 128)[0]                  # min-pool, f1tenth.py
      state = np.asarray(obs['state'], np.float32)                 # (5,) 이미 정규화
      return {0: np.concatenate([lidar128, state])}                # (133,)
  ```
- **★★ action raw↔env (정규화 충돌 해결)**: GuidedPolicy는 raw[steer rad, speed m/s] 출력, 그러나
  make_env 체인의 NormalizeActions는 [-1,1] 입력 기대(raw로 affine 역매핑). → **raw를 [-1,1]로 재정규화**해
  통과(권장). NormalizeActions 역식(`original=(a+1)/2*(high-low)+low`의 역):
  ```python
  S_MIN,S_MAX,V_MIN = -0.4189,0.4189,-5.0
  def raw_to_norm(raw, v_max=5.0):
      a = np.empty(2, np.float32)
      a[0] = 2*(raw[0]-S_MIN)/(S_MAX-S_MIN) - 1            # steer
      a[1] = 2*(raw[1]-V_MIN)/(v_max-V_MIN) - 1            # speed
      return np.clip(a, -1, 1)
  ```
  단 SelectAction(key="action")이 dict 기대 → `env.step({'action': raw_to_norm(action)})` 형식(eval_gate
  run_episode의 action dict 패턴 확인). 대안(b): f110_gym 내부 wrapper에 raw 직접(NormalizeActions 우회).
- **2랩 측정**: `info['cause']=='lap_complete'`(완주) + per-lap `obs['log_lap_time_s']>0`(eval_gate
  run_episode/aggregate 그대로) → 2랩 lap time. cap-5 baseline 107.16s 대비(floor=cap-15 37.3s 초과 목표).

## 5. 구현 순서 (다음 세션)
1. **device 파라미터화**: policies.py:55 → self.device, arrays.py:6-7 → DIFFUSER_DEVICE 환경변수.
2. **normalizer 저장 글루**: train.py(+train_values.py) dataset 직후 pickle dump.
3. **plan_f1tenth.py 신설**: 로딩부 재사용(value+scale=0.0) + normalizer 로드 + env_obs_to_cond +
   raw_to_norm + run_episode(GuidedPolicy) + aggregate(2랩 lap_time) + cap-5 비교.
4. **검증(verifier)**: 학습 ckpt로 sim 1ep — action shape(2)/normalize 정합/2랩 완주/lap_time 측정.

## 6. risks (P5 workflow)
- value 부실 → MPC best 선택 왜곡(value 학습 품질이 MPC 직접 좌우). P4 value loss corr 모니터.
- latency 50Hz 초과 가능(20 diff×64 batch×2 guide) → 실측(sim이라 결과엔 무관, 보고서 기록).
- normalizer.pkl scipy 버전 의존(언피클 실패 시 mins/maxs numpy 경량 덤프로 대체).
- check_compatibility는 normalizer 타입만 검사(통계 미검사) → 고정 pickle로 보강.
- action dict 형식(SelectAction key='action') — eval_gate run_episode의 정확한 step 인자 확인 필수.

---

## ★ 정정 (2026-06-20, 적대적 구현 검수 반영 — 본 절이 §4·§2 해당부분을 supersede)

> 구현 critic(새 세션) + 내 1차소스 재검증(코드 round-trip + npz 실측)으로 아래를 정정한다.
> 본 절이 위 §4(v_max)·§2(선택 기제)의 해당 부분보다 **우선**. 나머지(§1·§3·obs변환·normalizer 글루·§5·§6)는 유효.

### 1. [Critical] eval env **V_MAX = 20** (§4의 v_max=5.0 폐기)
- §4 line 65 `config.v_max = 5.0` + line 80 `raw_to_norm(..., v_max=5.0)`은 **틀렸다.** 그대로 구현하면
  raw_to_norm이 5 m/s 초과 모든 명령을 norm>1 → `np.clip(1)` → env(high=5)서 5 m/s로 매핑 →
  **차량이 5 m/s에 하드캡 = baseline cap-5와 동급 → floor/목표 달성 원천 불가.**
- 근거(npz 실측): Diffuser는 cap-20 데이터로 raw speed를 **최대 20 m/s까지** 학습(cap20 speed_norm∈
  [-0.889,1]·v_max=20 → raw ∈ [-3.6, 20]). eval env가 그 표현범위를 허용해야 한다.
- **수정(= SSOT 008 S4)**: eval env를 **V_MAX=20**(미수정 기본 env)으로 build + `raw_to_norm(raw, v_max=20)`.
  round-trip 검산: raw 12 → norm 0.36 → env(20) → 12 ✓ / raw 20 → norm 1 → 20 ✓. (per-tier v_max 금지.)
  baseline(107.16s) 비교는 *결과 lap time*으로 하는 것이지 *action space*를 5로 맞추는 게 아니다.

### 2. [정정] §2 line 33 "value sort 후 best 1개"는 코드와 불일치
- 실제 [policies.py:36] `action = actions[0,0]` — batch 64 중 **0번째 trajectory의 첫 action**을 그대로
  실행(argmax-over-value 선별 **없음**). 64개는 전부 value-gradient로 유도될 뿐 best 선별 단계는 없다.
- 함의: **value guidance gradient가 품질의 *유일* 레버**(선별 fallback 없음) → D3(value 품질)가 더 중요.
  batch_size는 1로 줄여도 결과 동일. 정확한 기제 = [[understand/001-diffuser-vs-value-architecture]].

### 3. normalizer 고정 — v1 운용
- §3 pickle 글루는 **현재 학습 중인 diffusion run엔 미적용**(train.py 이미 실행 중, 재읽기 없음). → **v1 eval은
  frozen 데이터 재fit에 의존**: `F1TENTH_DATA_DIR`(=crash_data 767ep)·`F1TENTH_LIDAR_DOWNSAMPLE=128`
  동결 시 SafeLimits 재fit이 결정론적 동일(검수 확인). **추가 수집 금지.** value v2 재학습 시 §3 글루 적용 권장.

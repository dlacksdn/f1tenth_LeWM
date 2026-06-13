# 005 — Diffuser 적응 touch-point 상세 (글루 코드 감사)

> 2026-06-13. 병렬 조사 워크플로우의 diffuser-glue 에이전트 결과를 보존. Diffuser를 D4RL 대신
> f1tenth 에피소드 npz로 학습/평가하도록 적응할 때의 구체적 코드 수정 지점. plan_new/003 D5의
> 근거. **코어(temporal.py/diffusion.py/helpers.py/training trainer)는 무변경.** append-only.

---

## 0. 전체 구조: Config + pickle 패턴
모든 글루가 `utils.Config(클래스경로문자열 → importlib import → 인자 dict로 인스턴스화 + self를
pickle 저장)` 패턴. eval 때 `*_config.pkl`을 그대로 재실행해 객체 재구성.
- `diffuser/utils/config.py:6-19,21-37,64-68` — `import_class`가 문자열 로드, savepath tuple이면
  `pickle.dump(self)`, `__call__`이 `self._class(*args,**self._dict)`.

## 1. d4rl 의존 — 전수 차단 필요 (P0 첫 게이트)
`import d4rl`가 **모듈 로드만으로 강제 실행**되고 여러 import 체인에 박혀 있음:
- `diffuser/datasets/d4rl.py:23-26` — `import d4rl` (핵심 폭탄)
- `diffuser/datasets/sequence.py:6-9` — `from .d4rl import load_environment, sequence_dataset`
- `diffuser/datasets/__init__.py:1-2` — `from .sequence import *` + `from .d4rl import load_environment`
- `diffuser/datasets/preprocessing.py:7` — `from .d4rl import load_environment`
- `diffuser/utils/rendering.py:8` — `import mujoco_py`, `:15` `from ...d4rl import load_environment`;
  `utils/__init__.py:6`이 `from .rendering import *`라 **utils import 단계에서 사망 가능**
- **권고**: d4rl.py의 `import d4rl`를 try/except graceful degrade. 한 곳만 패치하면 다른 체인에서
  여전히 죽으므로 전수 차단.

## 2. D4RL API 의존부 (통째 대체 대상)
- `d4rl.py:31-40` `load_environment`, `:42-53` `get_dataset`(env.get_dataset()+antmaze 분기),
  `:55-103` `sequence_dataset`(rewards/terminals/timeouts, env._max_episode_steps, env.name) —
  전부 D4RL OfflineEnv API. → **`diffuser/datasets/f1tenth.py` 신설**로 대체.
- `sequence.py:21-27` — `SequenceDataset.__init__`이 `load_environment(env)` + `env.seed()` +
  `sequence_dataset(env, preprocess_fn)` 호출. **여기가 핵심 스텁 지점**.
- `sequence.py:29-42` — ReplayBuffer 누적→finalize→DatasetNormalizer→make_indices→normalize().
  **에피소드 dict 키 계약**: observations/actions/rewards/terminals[/timeouts].
- `datasets/buffer.py:65,72-80` — `add_path`가 termination_penalty 시
  `assert not path['timeouts'].any()` → **timeouts 키 없으면 KeyError**. `buffer.py:12` `np.int`는
  numpy≥1.24에서 제거 → **`np.int64`로 수정**.

## 3. 신설 로더 어댑터 (신규 코드)
`diffuser/datasets/f1tenth.py`:
- `load_f1tenth_environment(npz경로)`: 가벼운 더미 env 클래스 (`.seed`=no-op, `._max_episode_steps`,
  `.name` 속성만) — sequence.py가 그대로 동작.
- `f1tenth_sequence_dataset(...)`: 에피소드별 `{observations, actions, rewards, terminals[, timeouts]}`
  dict를 yield. d4rl.py:55-103과 동일 시그니처/yield 계약. npz 키 매핑: lidar+state(또는 피처)→
  observations, action→actions, reward→rewards, is_terminal/is_last→terminals.

## 4. normalizer 영속화 (숨은 핵심 계약)
- **별도 저장 안 됨**. eval 때 `serialization.py:36-60 load_diffusion`이 `dataset_config(seed=seed)`로
  **데이터를 재로딩해 normalizer 재계산**(`:47`). `training.py:141-161 save/load`는 model/ema만 저장.
- `serialization.py:67-79 check_compatibility`는 normalizer **'타입'만** assert(통계값 미검사) →
  통계 미세변동을 못 잡음.
- **리스크**: 데이터 순서 비결정/이동 시 학습·평가 normalizer 불일치 → action unnormalize 왜곡.
- **권고**: 데이터 경로 고정+결정적 로딩 보장(최소안), 또는 Trainer.save에 normalizer pickle 덤프
  추가 + load_diffusion/GuidedPolicy에서 우선 로드(권장).

## 5. 렌더러 스텁
- `training.py:73-78,128-132,167-227` — `sample_freq>0`이면 render_reference/render_samples가
  `renderer.composite`+`normalizer.unnormalize` 호출.
- `rendering.py:8 import mujoco_py`, `:58-71` gym.make+env.sim 의존.
- **권고**: `NullRenderer`(composite/render_* no-op, `__init__(env)` 무시) 신설 + config renderer
  교체 + **train config `sample_freq=0`**. rendering 외부 파일에 두어 utils import 사망 회피.

## 6. 평가 루프 재작성 (plan_guided 대체)
- `scripts/plan_guided.py:23-42` 로딩부(load_diffusion×2 + check_compatibility + ValueGuide) **재사용**.
- `:53-69` policy_config(GuidedPolicy, guide, scale, normalizer=dataset.normalizer, sample_fn=
  n_step_guided_p_sample, n_guide_steps, t_stopgrad) **재사용**.
- `:76-118` env 루프(env.reset/step/get_normalized_score/state_vector, logger.render)는 **D4RL gym
  의존 → f1tenth sim 루프로 전면 교체**(`scripts/plan_f1tenth.py` 신설).
- `sampling/policies.py:23-42 GuidedPolicy.__call__`: preprocess→정규화→diffusion_model(cond,guide)→
  actions=traj[:,:,:action_dim]→unnormalize→**actions[0,0]만 실행(MPC 암묵 재계획)**.
  `policies.py:55 device='cuda:0' 하드코딩` → 파라미터화.
- `helpers.py:142-145 apply_conditioning`: `conditions={time:value}` dict, observation 차원에만 적용.
  GuidedPolicy의 `{0: obs}`와 일치 → **현재 관측만 `{0: obs}`로 주면 됨**.
- `diffusion.py:159-193`: `batch_size=len(cond[0])`, `shape=(batch,horizon,transition_dim)`. cond[0] 필수.

## 7. value 학습 (거의 무변경)
- `scripts/train_values.py:20-33` — value dataset_config에 discount/termination_penalty/normed 추가,
  loader='datasets.ValueDataset'. 나머지 train.py와 동일.
- `sequence.py:106-148 ValueDataset` — `fields['rewards'][path_ind,start:]`로 discounted return-to-go
  계산. **rewards 필드만 있으면 동작**. normed=True면 _get_bounds 전체 스캔.
- value guidance 미사용이면 train_values/plan_guided/ValueGuide 전부 생략하고 unconditional plan 가능.

## 8. 모델 차원 자동 전파 (주의)
- `train.py:39-40,51-52,62-63` — observation_dim/action_dim을 dataset 속성(sequence.py:37-38,
  `fields.observations.shape[-1]`)에서 읽어 transition_dim=obs+act, cond_dim=obs 자동 설정.
- **리스크**: npz obs 차원이 에피소드마다 다르면 `buffer._allocate`(buffer.py:57-61)에서 깨짐 →
  **obs 차원 일관성 사전 검증 필요**.

## 9. config 신설
- `config/f1tenth.py`: base['diffusion'/'values'/'plan'], loader='datasets.F1tenthSequenceDataset',
  renderer='utils.NullRenderer', normalizer(라이다 고정범위면 SafeLimitsNormalizer
  `normalization.py:177` 권장 — 상수 차원 안전), horizon/max_path_length를 에피소드 길이에 맞춤.
- `utils/setup.py:64-85 read_config` — `args.dataset.replace('-','_')`로 override 매칭.

## 공수 추정 (구현 게이트에서 확정)
로더 어댑터+sequence 패치+d4rl 차단 0.5~1일 / NullRenderer+config 0.5일 / normalizer 영속화 0.5일(선택) /
평가 루프 재작성 1~2일(f1tenth sim 복잡도 의존) / value 배선 0.5일. **코어 무변경.**

## 미검증 (구현 게이트)
- f1tenth npz 실제 키/shape/에피소드 길이 분포 → buffer 계약·max_path_length 결정 (P0/P3에서 1개 로드 확인)
- normalizer 선택(LimitsNormalizer vs Gaussian vs SafeLimits) → 데이터 분포 의존
- termination_penalty 정책(충돌 반영) → buffer assert 회피 설계
- value guidance 사용 여부 → 미사용 시 배선 대폭 생략

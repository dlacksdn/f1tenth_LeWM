# 001 — Diffuser 아키텍처: diffusion 모델과 value의 역할 (엄밀)

> 2026-06-19. "디퓨저가 월드모델이고 value가 action을 선택하나?"라는 질문에 대한 1차소스(repo
> 코드) 기반 정밀 설명. 목적 = 타 세션 인수인계 + 개념 정합. 근거 코드:
> `vendor/diffuser/diffuser/sampling/{policies,functions}.py`, `config/f1tenth.py`.
> 관련: [[008-diffuser-plan-v4]](성공기준), Dreamer(=RL_project, 사용자가 직접 구현한 월드모델)와 대조.

---

## 0. 한 줄 정답
- **학습 순서**: diffusion 먼저 → value 뒤. 단 **의존성 없음**(독립 네트워크, 같은 데이터). 순서는 **GPU 8GB
  제약**(각 ~7GB라 동시 불가)일 뿐.
- **디퓨저는 월드모델이 아니라 "궤적 생성기(trajectory generator)"**. value는 action 선택기가 아니라
  **critic**으로, 그 **gradient가 궤적 생성을 고-return 방향으로 유도(guide)**한다. action은 "유도되어 생성된
  궤적의 첫 step"을 실행한다.

---

## 1. 두 네트워크는 독립적이고, 순서는 메모리 때문

Diffuser는 **두 개의 별도 신경망**을 **같은 데이터**로 따로 학습한다.

| 네트워크 | 클래스 | 학습 목적 | reward 필요? |
|---|---|---|---|
| **diffusion 모델** | `TemporalUnet` + `GaussianDiffusion` | 궤적 분포 `p(τ)` denoising | ❌ (관측+행동만) |
| **value 모델** | `ValueFunction` + `ValueDiffusion` | 궤적의 return-to-go 회귀 | ✅ (reward로 return 계산) |

- value 학습은 **학습된 diffusion 모델을 전혀 쓰지 않는다** — 데이터에서 직접 계산한 discounted
  return-to-go를 타깃으로 회귀할 뿐이다.
- 따라서 `diffusion 먼저 → value 뒤`는 **성능 의존이 아니라 GPU 메모리 직렬화**다(8GB에 ~7GB×2 동시 불가).
  순서를 바꿔도 무방. 평가 때 둘을 같이 로드해 결합한다.

---

## 2. 디퓨저 = 월드모델이 아니라 **궤적 생성기**

### 2.1 월드모델(Dreamer)과의 결정적 차이

| | **Dreamer (월드모델, model-based RL)** | **Diffuser** |
|---|---|---|
| 학습 대상 | `p(s_{t+1}\|s_t,a_t)` = **1-step 동역학** + reward | `p(τ)` = **궤적 전체 분포**, state·action을 *동시에* |
| τ의 정체 | — | `[(obs, action)]×horizon`, transition_dim=**135** = obs **133** + action **2** |
| 계획 방식 | 모델을 **롤아웃**(step별 시뮬)하며 행동 탐색 | 궤적을 **통째로 sampling**(reverse diffusion) — **생성 자체가 계획** |
| 행동 출처 | 별도 **actor 네트워크**가 출력 | 생성된 궤적에서 **직접 읽음** (actor 없음) |
| 동역학 표현 | 명시적(다음 상태 예측) | **암묵적**(샘플 궤적의 state 전이가 데이터와 정합) |

### 2.2 정확한 표현
디퓨저는 동역학을 *암묵적으로만* 담는다(생성된 궤적의 state 전이가 데이터 분포와 맞으므로). 하지만
**step-by-step 시뮬레이터로 쓰지 않고**, 한 번에 horizon 128(=2.56s) 길이의 (관측,행동) 시퀀스를
denoising으로 생성한다. 그래서 "월드모델 풍미"는 있으나, 정밀하게는 **"state+action 시퀀스에 대한 생성
prior"**다. (보고서 표기는 "diffusion 기반 offline planning"이 정직 — world model 정체성은 부차적.)

> 비유: 디퓨저 = **학습된 궤적 최적화기(trajectory optimizer)**. diffusion = "무엇이 실현가능·그럴듯한가"라는
> feasibility/behavior prior, value = "어느 방향이 더 좋은가"라는 목적함수. guided sampling으로 둘을 결합.

---

## 3. value는 action을 "선택"하지 않는다 — 생성을 **유도**한다

value는 actor가 아니라 **critic**이다. 행동 후보 중에서 고르는 게 아니라, **궤적의 예측 return을 매기고
그 gradient로 denoising을 고-return 방향으로 미는** 역할(= classifier-guided diffusion). 코드가 정확히 그렇다.

### 3.1 매 denoising step ([sampling/functions.py:9-35] `n_step_guided_p_sample`)
```python
for _ in range(n_guide_steps):                 # config: n_guide_steps=2
    y, grad = guide.gradients(x, cond, t)       # y=value 예측, grad=∇(value)/∇(궤적 x)
    if scale_grad_by_std: grad = model_var * grad
    grad[t < t_stopgrad] = 0                     # 마지막 몇 step은 유도 중단(t_stopgrad=2)
    x = x + scale * grad                         # ★ 궤적을 고-value 쪽으로 밀기 (scale=0.1)
    x = apply_conditioning(x, cond, model.action_dim)   # 현재 obs(위치0) 고정
model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)  # diffusion denoise
return model_mean + model_std * noise, y
```
→ 각 denoising step = **(a) value gradient로 고-return 방향 밀기 → (b) diffusion 모델로 denoise**. 20번의
diffusion timestep 내내 반복.

### 3.2 한 step 행동 결정 ([sampling/policies.py:23-42] `GuidedPolicy.__call__`)
```python
conditions = {0: 현재 obs}                                  # 지금 관측을 궤적 시작점(위치0)에 고정
samples = self.diffusion_model(conditions, guide=self.guide, **sample_kwargs)  # value-유도 reverse diffusion
trajectories = to_np(samples.trajectories)                  # (batch, horizon, 135)
actions = trajectories[:, :, :self.action_dim]              # 궤적의 action 부분 (앞 2D)
actions = self.normalizer.unnormalize(actions, 'actions')   # raw [steer rad, speed m/s]로 환원
action  = actions[0, 0]                                      # ★ 첫 trajectory의 첫 step만 실행
```
- **action은 "value가 유도해 생성된 궤적의 첫 step"**이다. value가 따로 action을 출력하거나 후보 중 고르는
  게 아니다(여기엔 argmax-over-samples도 없음 — 순수 gradient 유도 후 `[0,0]`).
- `action[0,0]`만 실행하고 매 step 다시 계획 = **K=1 MPC**(매 step 재계획). open-loop drift 없음.

---

## 4. 역할 분담 요약

- **diffusion 모델** = "무엇이 그럴듯·실현가능한 주행인가" — feasibility prior + 행동 prior + 암묵 동역학.
  (in-distribution 보장: 데이터에 없는 궤적은 잘 안 만든다 → 속도 외삽 불가의 근거.)
- **value 모델** = "어느 방향이 더 빠르고 안전한가" — 목적함수(critic), gradient로 생성을 유도.
- **합** ≈ `argmax_τ value(τ)  s.t.  τ가 데이터상 그럴듯` 을 **gradient-guided sampling으로 근사 탐색**한 뒤
  **첫 action 실행 + 재계획(MPC)**.

Dreamer 대비 가장 큰 차이: **별도 actor(정책망)가 없다.** "정책"이라는 게 곧 "value로 유도해 궤적을 생성하고
첫 action을 실행"하는 **절차 자체**다.

---

## 5. 보충 (엄밀성)

- **왜 value가 "ValueDiffusion"인가**: 평범한 MLP value가 아니라, *노이즈 낀* 궤적의 return을 noise level `t`별로
  예측하도록 학습된다. 그래야 denoising 도중(아직 노이즈가 있는 `x`)에도 gradient가 의미를 갖는다. 그래서
  `guide.gradients(x, cond, t)`에 `t`가 들어간다.
- **conditions의 역할**: `{0: 현재 obs}`는 "궤적의 0번째 timestep 관측은 현재 상태로 고정"이라는 제약. 매
  denoising step 후 `apply_conditioning`으로 다시 박아, 생성 궤적이 "지금 여기"에서 출발하도록 한다.
- **transition 구성**: 각 timestep = concat(action(2), observation(133)) = 135D. `policies.py`가 앞 2D를 action,
  나머지 133D를 observation으로 분리한다. → **이 구현은 state와 action을 *함께* 생성**(원조 Diffuser 방식;
  inverse-dynamics로 action을 따로 빼는 Decision Diffuser 변형이 아님).

---

## 6. 코드 위치 레퍼런스
- 유도 sampling 루프: `vendor/diffuser/diffuser/sampling/functions.py:9-35`
- 정책(행동 결정): `vendor/diffuser/diffuser/sampling/policies.py:23-42` (첫 action = `:36`)
- 하이퍼파라미터(scale/n_guide_steps/t_stopgrad/discount): `vendor/diffuser/config/f1tenth.py` `plan` 블록
- value 모델: `vendor/diffuser/diffuser/models/temporal.py` `ValueFunction`(off-by-one fork-patch 적용)
- diffusion 모델: 동 파일 `TemporalUnet`, `diffuser/models/diffusion.py` `GaussianDiffusion`/`ValueDiffusion`

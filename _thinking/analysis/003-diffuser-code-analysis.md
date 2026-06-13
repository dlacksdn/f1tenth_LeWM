# 003 — Diffuser 코드 분석: f1tenth 적용 가능성 확정

> 2026-06-13. 모델을 LeWM(JEPA) → **Diffuser**(Planning with Diffusion for Flexible Behavior
> Synthesis, Janner et al. ICML 2022)로 전환 결정 후, 공식 코드(`~/planning_with_diffusion`,
> github.com/jannerm/diffuser clone)를 정독해 "정말 f1tenth에 써도 되는지"를 코드 근거로
> 확인한 기록. append-only.

---

## 배경 (이 문서만 읽어도 되도록)

**Diffuser란**: 궤적 τ=(state, action 시퀀스) 전체를 **diffusion 모델로 생성**하는 offline
planner. 학습은 offline 데이터로만(환경 상호작용 0). 주행 시에는 "현재 관측을 조건으로 미래
궤적을 생성 → 첫 action 실행 → 재생성(MPC)". "빠르게"는 **별도 value 함수가 sampling을
고-return 궤적 쪽으로 유도(guidance)**해서 달성. 과제(100초대 policy 데이터 → offline RL로 더
빠른 policy)에 구조적으로 맞음.

**결론 먼저**: **f1tenth에 쓸 수 있다(✅). 단 4가지 적응 작업이 필요**하고, 그중 1개(lidar
차원)가 유일한 실제 설계 리스크다. 핵심 모델 코드는 순수 torch라 무거운 의존성(d4rl/mujoco)
없이 돌릴 수 있다.

---

## 1. 데이터 계약 (무엇을 먹는가) — 매우 단순함

Diffuser가 요구하는 입력은 **에피소드 dict의 iterator**, 키 4개뿐
([datasets/d4rl.py:55-99](file:///home/dlacksdn/planning_with_diffusion/diffuser/datasets/d4rl.py) `sequence_dataset`):
```
observations, actions, rewards, terminals
```
- 우리 npz가 이미 이 형태를 가짐: observations=(lidar+state), actions=(steer,speed), reward,
  is_terminal/is_last. → **d4rl 로더만 우리 npz 로더로 교체하면 됨** (~50줄)
- `SequenceDataset`([datasets/sequence.py:82-91](file:///home/dlacksdn/planning_with_diffusion/diffuser/datasets/sequence.py)):
  각 datapoint = `trajectory = concat([actions, observations], axis=-1)`, shape
  **(horizon, action_dim + obs_dim)**. 정규화는 LimitsNormalizer(obs·action).
- planning용 조건: `get_conditions → {0: observations[0]}` = **현재 관측 1개를 첫 timestep에
  고정**(sequence.py:73-77).

## 2. 모델 구조 = 시간축 1D U-Net (lidar 차원 문제의 근원)

[models/temporal.py:49-146](file:///home/dlacksdn/planning_with_diffusion/diffuser/models/temporal.py)
`TemporalUnet`:
- 입력 (batch, horizon, transition_dim)을 `rearrange 'b h t -> b t h'`(line 120) →
  **transition_dim을 "채널"로, horizon을 "공간축"으로** 1D conv.
- 즉 `transition_dim = action_dim(2) + obs_dim`. 우리가 obs에 raw lidar(1080)+state(5)를
  그대로 넣으면 **transition_dim = 1087 채널**.

⚠️ **이게 유일한 실제 리스크 (#1 설계 결정)**:
1. 1087 채널은 무겁고, 1080 lidar의 **공간 구조(인접 빔 상관)를 채널로 펴버려 낭비**.
2. 더 본질적: Diffuser는 **미래 observation도 생성**한다(궤적의 일부). 즉 모델이 **미래 lidar
   스캔 1080차원을 매 step 상상**해야 함 — D4RL의 저차원 proprioceptive state(11~39차원)보다
   훨씬 어려운 생성 과제.
3. → **lidar를 저차원으로 축소**해야 함. 옵션:
   - (a) **단순 다운샘플** 1080 → ~64~108 빔 (구조 보존, 학습 0, 결정적·가역) ← v1 1순위 추천
   - (b) RL_project 학습 encoder(ConvEncoder1D 1080→512, [[004-dreamer-reuse...]]) — 512는
     diffusion 채널로 과대, 재학습해 축소 필요
   - (c) centerline 기반 피처(중심선까지 거리, heading 오차, 전방 곡률, 속도)로 저차원
     state 구성 — 가장 가볍지만 피처 엔지니어링 필요
   - 주의: 우리는 미래 lidar를 **디코딩하지 않음**(생성 궤적에서 action만 사용 →
     [sampling/policies.py:36](file:///home/dlacksdn/planning_with_diffusion/diffuser/sampling/policies.py)).
     그래도 모델이 latent/축소-lidar를 **생성**은 해야 하므로, 축소 표현이 모델링하기 쉬워야 함.

## 3. "빠르게" 메커니즘 = value guidance (과제 핵심)

- **value 정의**: `ValueDataset`([sequence.py:138-147]) value = Σ γ^t · reward = **할인 누적
  보상**. 우리 reward(progress 중심)를 그대로 사용 → value가 곧 "주행 진행 빠름".
- **value 모델**: `ValueDiffusion`/`ValueFunction`([diffusion.py:235-249], [temporal.py:149-235])를
  **diffusion 모델과 별도로 학습**(scripts/train_values.py).
- **유도**: `GuidedPolicy`([policies.py:23-42]) + `ValueGuide.gradients`([guides.py:16-21])가
  denoising 매 step마다 ∇_x value로 샘플을 고-value 쪽으로 밀어줌 → **고-return(빠른 진행)
  궤적 생성** → 첫 action 실행.
- ⇒ **학습 대상 2개**: ① unconditional 궤적 diffusion(train.py) ② value 함수(train_values.py).
  둘 다 같은 offline 데이터로. 이게 Diffuser 정본 레시피.
- ⇒ behavior policy보다 빨라지는 원리: value guidance가 **데이터에 있는 빠른 구간들을
  재조합(stitching)** 해 평균보다 높은 return 궤적을 합성. 과제의 "기존 policy 대비 개선"과 정합.

## 4. 의존성 — 무거운 건 다 버릴 수 있음

[environment.yml](file:///home/dlacksdn/planning_with_diffusion/environment.yml): gym 0.18,
mujoco-py, d4rl, **torch 1.9.1+cu111**, jax/flax, ray 등 — 전부 **D4RL 벤치마크 인프라용**.
- 핵심 모델 코드(temporal.py / diffusion.py / helpers.py)는 **순수 torch + einops**.
- datasets/d4rl.py만 d4rl import → **우리 npz 로더로 교체하면 d4rl/mujoco/jax 전부 불필요**.
- → **기존 RL_project venv(py3.8, torch 2.4.1)에서 핵심 코드 구동 가능** (1.9→2.4 API 드리프트는
  conv1d/einops/autograd 수준이라 사소). 또는 가벼운 전용 venv. **무거운 설치 없음.**

## 5. 평가 통합 — LeWM의 C-1 문제가 사라짐

- 관측이 **네이티브 lidar+state**(축소만) → **렌더링 파이프라인 불필요**.
- 최종 정책 = `GuidedPolicy`(torch 추론) → **f110_gym이 도는 py3.8 RL_project venv에서 그대로
  추론** 가능 → LeWM 때의 크로스-프로세스/py3.10 문제 소멸. 평가가 오히려 단순.
- planning이 sampling이라 wall-clock 느림 → **sim 시간 = lap time**이므로 매 plan마다 sim을
  멈췄다 진행하면 무방(보고서에 명시).

---

## 6. 필요한 적응 작업 (구현 항목)

| # | 작업 | 난이도 | 비고 |
|---|---|---|---|
| 1 | **lidar 차원 축소** (다운샘플/인코딩 → ~64~108) | 중 | **유일한 실제 리스크**. v1=단순 다운샘플 |
| 2 | f1tenth npz 로더 (d4rl.py 대체, sequence_dataset 포맷) | 하 | ~50줄, 키 4개 |
| 3 | value 함수 학습 배선 (progress reward → 할인 return) | 하 | 정본 train_values.py 적응 |
| 4 | 의존성 디커플 + GuidedPolicy를 f1tenth env에 연결 | 중 | py3.8 venv 재사용 |

## 7. 미검증 / 다음 분기에 확인

- torch 1.9→2.4 실제 구동 시 사소한 deprecation(예: `torch.cumprod` 인자, buffer dtype) —
  smoke로 확인
- 축소 lidar를 diffusion이 잘 **생성**하는지(미래 lidar 상상의 난이도) — 파일럿 학습 후
  생성 궤적 품질 점검 필요. 이게 안 되면 (c) 피처 기반 state로 전환
- horizon·transition_dim 조합에서 8GB 메모리 — 파일럿에서 batch 스윕

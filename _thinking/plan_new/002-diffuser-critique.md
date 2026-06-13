# 002 — Diffuser offline RL 계획(001) 비판적 검토 (critic)

> 2026-06-13. [[001-diffuser-project-plan]] v1 계획에 대한 read-only critic 검토.
> 1차 소스(Diffuser 코드 `~/planning_with_diffusion`, RL_project 스냅샷·metrics·npz)를 직접
> 열어 사실 검증. 어떤 파일도 수정하지 않음. append-only.
> 선행: [[003-diffuser-code-analysis]], [[004-dreamer-reuse-and-behavior-policy]], [[003-project-spec]].

---

## 0. 종합 평결: **수정 후 진행 (Revise-then-proceed)**

Diffuser 모델 선택과 코드 적용 가능성은 **건전하다** — 데이터 계약·value guidance·의존성 디커플
주장은 모두 1차 소스로 확인됐다(§1). lidar 차원(D1)도 관리 가능하다.

그러나 **계획이 "유일한 실제 리스크"로 지목한 것(D1 lidar)은 진짜 1순위 리스크가 아니다.**
실제 1순위는 **D2 — "~100초대 behavior policy"가 존재한다는 전제이고, 이것은 RL_project의
실측 학습곡선으로 반증된다.** 저장된 어떤 스냅샷도 한 바퀴를 완주하지 못하며(초기 체크포인트는
~3초 만에 크래시), 완주 능력은 step 111k 이후에야 ~18초대로 등장한다. **"100초대에 완주하는
policy"라는 중간 단계 자체가 이 학습 런에 존재하지 않는다.** 따라서 계획의 D2-방법A(초기
체크포인트 사용)와 폴백(warm-load 짧은 재학습)은 둘 다 100초 policy를 만들지 못한다 — 크래시
데이터만 나온다.

동시에, 계획의 "향상 vs 천장" 논리는 원리적으로 옳지만(§3), **이 데이터셋 설계와 결합하면
모순**이다: in-distribution 생성기인 Diffuser는 데이터에 없는 속도를 외삽하지 못한다. 균일하게
느린(혹은 크래시하는) 데이터만 모으면 "100초보다 빠른 주행"을 합성할 재료가 없다.

결론: 모델·코드 라인은 그대로 가되, **(1) behavior policy / 데이터 생성 전략을 "expert 속도캡
범위"로 재설계**하고, **(2) D1을 단순 다운샘플 대신 centerline-피처 저차원 state 우선으로
재고**하면 v1이 성공 기준("100초 baseline보다 빠른 2랩 완주")에 실제 도달할 수 있다. 현재 P0~P6
순서에는 빠진 단계(100초 policy를 *만드는* 단계, baseline lap time 실측, 추론용 normalizer
직렬화, MPC 재계획 주기)가 있다.

---

## 1. 사실 검증 — 계획·003/004의 코드 주장 (대부분 ✅, 정밀화 필요 몇 건)

### 1.1 데이터 계약 ✅ (정밀화 1건)
- `sequence_dataset`은 `observations / actions / rewards / terminals`를 요구
  ([datasets/d4rl.py:55-99](file:///home/dlacksdn/planning_with_diffusion/diffuser/datasets/d4rl.py)). 확인.
- `trajectory = concat([actions, observations], axis=-1)`, shape `(horizon, action_dim+obs_dim)`
  ([datasets/sequence.py:89](file:///home/dlacksdn/planning_with_diffusion/diffuser/datasets/sequence.py)). 확인.
- planning 조건 = `get_conditions → {0: observations[0]}` (현재 관측 1개를 t=0에 고정)
  ([sequence.py:73-77]). 확인.
- **정밀화**: `sequence_dataset`은 *flat* 데이터셋을 받아 `terminals`/`timeouts`로 에피소드를
  잘라 yield하는 구조다. 우리 npz는 **이미 에피소드 단위 파일**이므로, d4rl.py를 통째로
  대체하는 게 아니라 **"npz 파일 1개 → dict 1개"를 yield하는 더 단순한 itr**로 교체하면 된다
  (~30줄, 계획의 "~50줄"보다 쉬움). 우리 npz 키 매핑: `is_terminal`/`is_last` → `terminals`,
  `reward` → `rewards`, `lidar`+`state` → `observations`(concat), `action` → `actions`.
  ⚠️ 단 `SequenceDataset.__init__`이 `load_environment(env)` + `env.seed()`를 호출하므로
  ([sequence.py:22-23]) 이 두 줄도 stub해야 함(d4rl/gym import 회피의 일부). 계획이 "d4rl
  디커플"로 묶어 인지하고 있으나 **sequence.py 상단의 `from .d4rl import ...`도 손대야 함**을
  P0/P3에 명시할 것.

### 1.2 모델 = transition_dim을 채널로 쓰는 1D U-Net ✅
- `TemporalUnet.forward`: `einops.rearrange(x,'b h t -> b t h')`로 transition_dim을 채널,
  horizon을 공간축으로
  ([models/temporal.py:120](file:///home/dlacksdn/planning_with_diffusion/diffuser/models/temporal.py));
  `dims=[transition_dim, dim*1, dim*2, ...]`([temporal.py:62]). 확인.
- ⚠️ **메모리 주장 정정**: 계획·003은 "1087채널 / 8GB 메모리"를 리스크로 든다. 실제로는
  메모리가 binding이 **아니다**. dim=32, dim_mults(1,2,4,8), horizon 32, batch 32 기준 활성값은
  batch×1087×32 ≈ 1.1M float(4MB)급으로 raw lidar여도 8GB에 충분히 들어간다. **진짜 문제는
  메모리가 아니라 "미래 lidar 1080차원을 매 step *생성*하는 학습 난이도"**다(§2). 보고서/계획
  문구에서 "메모리 부담"을 "생성 학습 난이도"로 교정 권장.

### 1.3 value guidance ✅ (의미론 정밀화 1건 — 중요)
- value 정의: `ValueDataset.__getitem__` value = `(discounts * rewards).sum()`
  ([sequence.py:138-147]). 확인. 단 **`rewards = fields['rewards'][path_ind, start:]` — horizon
  창이 아니라 "start부터 에피소드 끝까지"의 할인 return-to-go**다. 즉 value는 짧은 horizon이
  아니라 *남은 에피소드 전체*의 누적 진행을 본다 → lap time 최적화에 오히려 유리(긴 신용할당).
- value 모델: `ValueDiffusion`/`ValueFunction`을 diffusion과 **별도 학습**
  ([diffusion.py:235-249], [temporal.py:149-235]). 확인.
- 유도: `n_step_guided_p_sample`이 매 denoising step에서 `guide.gradients`(∇_x value)로
  `x += scale·(model_var·grad)` ([sampling/functions.py:9-37]) +
  `ValueGuide.gradients`([sampling/guides.py:16-21]) + `GuidedPolicy`가 첫 action 추출
  ([sampling/policies.py:36]). 확인. 기본 하이퍼파라미터: `n_guide_steps=2, scale=0.1,
  t_stopgrad=2`([config/locomotion.py:119-121]).
- ⚠️ 정밀화: `p_mean_variance`에 `else: assert RuntimeError()`([diffusion.py:152])라는 죽은 코드가
  있다(예외 인스턴스는 truthy → assert 통과, no-op). locomotion은 `clip_denoised`/`predict_epsilon`
  설정에 의존하므로 우리 config에서 **clip_denoised=True 권장**(정규화 범위 [-1,1] clamp 필요).

### 1.4 의존성 디커플 ✅
- `environment.yml`의 d4rl/mujoco-py/jax/ray는 전부 D4RL 벤치마크용; 핵심
  temporal.py/diffusion.py/helpers.py는 순수 torch+einops. 확인.
- torch 1.9→2.4: `torch.cumprod`, `register_buffer`, `einops.rearrange`, `autograd.grad`,
  `nn.Mish` 모두 2.4 호환. `betas*np.sqrt(alphas_cumprod_prev)`([diffusion.py:84-87])처럼 np
  ufunc이 텐서에 섞여 들어가는 코드가 있어 smoke에서 dtype 확인 필요(사소). P0 유지.
- ⚠️ **미감사**: scripts/train.py, train_values.py, plan_guided.py의 **글루(Config 객체 구성,
  dataset/model 빌드 배선)는 라인별로 확인하지 못함**. 핵심 모델·샘플링·데이터 계약은 검증됐으나,
  학습 스크립트의 Config/utils 의존(`diffuser.utils`)을 우리 환경에 붙이는 작업이 P0~P4의 숨은
  공수다. 과소평가 금지(§확인 못 한 항목).

---

## 2. D1 (lidar 차원) — 권고: **단순 다운샘플 말고 centerline-피처 state를 1순위로**

계획은 (a) 단순 다운샘플(64~108)을 1순위, (c) centerline 피처를 최후순위로 둔다. **이 우선순위를
뒤집을 것을 권고한다.** 근거:

**(a) 다운샘플의 한계**
- 다운샘플은 채널 수를 줄여 메모리/연산은 낮추지만, **§1.2에서 보였듯 메모리는 애초에 문제가
  아니다.** 다운샘플이 실제로 낮춰야 하는 건 "생성 난이도"인데, 64~108차원 lidar라도 Diffuser는
  여전히 **미래의 벽거리 스캔 벡터 흐름을 생성**해야 한다(궤적의 일부). 이건 D4RL의 저차원
  proprioceptive state(11~39)보다 본질적으로 어렵다 — lidar는 차량 위치/자세에 비선형적으로
  의존하는 고도로 구조화된 신호다.
- Oschersleben 기하 보존 측면: 270° FOV 기준 64빔 ≈ 4.2°/빔 → 벽·갭 거리엔 충분하나, 자기교차
  고속 트랙의 미세한 라인 차이를 표현하긴 빈약할 수 있음. 보존성과 생성용이성이 트레이드오프.

**(c) centerline-피처 state — 이 과제(단일·기지 트랙)에 최적**
- 우리는 **Oschersleben centerline.csv를 이미 보유**하고, env가 info에 `arclen_s`/`closest_idx`를
  제공한다([[004-dreamer-reuse-and-behavior-policy]] §5). 따라서 다음과 같은 **~8~14차원 저차원
  state**를 결정적으로 계산 가능:
  `[중심선 횡오차 e_y, heading 오차 e_ψ, 전방 곡률 κ(s+L1·L2·L3 lookahead 3~5개), 현재 속도 v,
  yaw rate, slip(=vel_y/vel_x), prev_steer, prev_speed]`.
- 장점: ① Diffuser가 **생성하기 쉬운** 매�us러운 저차원 dynamics(곡률은 트랙 함수 → 거의
  결정적), ② **Markov 충분**(차량 동역학 + 전방 기하), ③ value/progress와 **직접 정렬**
  (e_y·κ·v가 곧 "빠른 진행"), ④ lidar 생성 난이도 리스크(계획의 리스크 #1) **완전 소거**.
- 비용: 피처 엔지니어링 + 추론 시 env obs로부터 온라인 피처 계산(closest_idx/arclen으로 가능,
  곡률은 centerline에서 사전계산 테이블). 단일 트랙이므로 트랙 일반성 손실은 무관(과제는
  Oschersleben 단일).
- **이것이 racing RL의 표준 state 설계**이며, "lidar 미래 생성"이라는 가장 큰 미검증 리스크를
  피하는 가장 저렴한 길이다.

**권고 (D1 재정의)**
- **v1 1순위: centerline-피처 저차원 state(~10차원) + state[v, yaw_rate, slip, prev_a].**
  transition_dim = action 2 + 피처 ~10 = ~12. D4RL hopper(11) 급 → Diffuser가 가장 잘 다루는
  영역. 생성·메모리·학습 모두 쉬움.
- **v1 병행/대안: 다운샘플 lidar 32~64 + 위 피처 일부(하이브리드).** lidar를 "모델-프리 백업"으로
  남기되, 순수 lidar 단독은 비권장.
- 파일럿(P4)에서 두 표현의 **생성 궤적 품질**(미래 state 매끄러움·물리 타당성)을 비교해 확정.
- ⚠️ 주의: 어떤 표현을 쓰든 **수집 npz에 raw lidar+state를 그대로 저장**해 두고, 로더에서 피처로
  변환하라(표현 변경 시 재수집 불필요). 데이터 수집(P2)과 표현 결정(P3)을 디커플.

---

## 3. "향상 vs 천장" — 원리 맞음, **데이터 설계와 결합 시 모순**, 보강 필수

계획의 주장 "behavior 속도엔 천장이 없지만 데이터 커버리지엔 천장이 있다"는 **Diffuser의 실제
동작과 정합**하다:
- value guidance + stitching이 데이터의 좋은 구간을 재조합 → 단일 trajectory보다 높은 return
  합성(D4RL 검증 원리). ✅
- diffusion은 **in-distribution** 생성기 → 데이터에 없는 dynamics는 외삽 못 함(offline 안전장치). ✅

**그러나 비판점 — 계획의 데이터 설계가 이 천장을 스스로 낮춘다:**
1. Diffuser는 **데이터에 존재하는 속도/라인만 재조합**한다. behavior policy가 균일하게 100초대로만
   달리면, "100초보다 빠른 궤적"을 만들 **원재료(빠른 구간)가 데이터에 없다** → value guidance가
   끌어올릴 천장이 ≈100초. 미세한 라인 최적화로 몇 초 깎는 수준에 그칠 위험.
2. 계획의 다양성 확보책("여러 초기 체크포인트 + action noise")은 **§4의 실측으로 무효**다 — 초기
   체크포인트는 빠른 구간이 아니라 **크래시(짧은 에피소드)**를 만든다. noise도 크래시를 늘릴 뿐
   "빠르고 안전한" 재료를 만들지 못한다.

**보강 (핵심 설계 수정):** 데이터 다양성은 **expert(16~19초)에 속도캡을 단계적으로 적용**해 만든다.
예: 속도 스케일 {×0.16→~100s, ×0.27→~60s, ×0.4→~40s, ×0.65→~25s, ×1.0→~17s}로 각각 rollout.
- 모든 캡이 **트랙을 완주**(expert 라인 유지) → 풀-트랙 커버리지 확보.
- 속도 스펙트럼 전체가 데이터에 존재 → value guidance가 "빠른 구간"을 재조합할 재료 확보 →
  100초보다 빠른 합성이 **구조적으로 가능**.
- 과제 정합: **"behavior policy = ×0.16 캡(≈100초)"를 baseline으로 명시**하고, Diffuser 정책이
  그보다 빠른 lap을 내면 "기존 policy 대비 개선" 데모 성립. (데이터에 더 빠른 캡이 섞였다는 점은
  보고서에 투명하게 기술 — offline RL은 수집된 데이터 전체를 쓰는 게 정상이며, 이는 "환경 추가
  상호작용 없음" 제약을 위반하지 않는다. 다만 "단일 behavior policy"라는 순수성에서는 타협.)

**향상 폭 현실 추정:** 데이터에 ×1.0(≈17초) 라인이 포함되면, Diffuser+guidance는 **데이터 최속에
가깝되 그보다 약간 아래/위(stitching·라인 합성)** 를 낼 가능성이 높다 → 보수적으로 **25~45초대,
잘 되면 17~25초대**. 100초 baseline 대비 개선은 **명확히 달성 가능**. 반대로 데이터를 100초
단일캡으로만 모으면 **개선 ≈ 0~10% (90~100초)** 에 그칠 위험이 크다. **데이터 다양성이 향상
폭을 사실상 결정한다** — 이 점은 계획도 말하지만, *어떻게* 다양성을 만들지가 틀렸다.

---

## 4. D2 (behavior policy 실현성) — **[Critical] 전제가 데이터로 반증됨**

`runs/stage2_oschersleben/`에 step_5k~step_85k 인터벌 스냅샷이 **모두 존재**(154MB 풀-에이전트)
+ `policy_best_lap16.6s_step82k/85k.pt`(48MB 추론 policy) 존재. **여기까진 계획대로.**

그러나 **`metrics.jsonl` 실측이 D2의 핵심 전제를 무너뜨린다:**

**(증거 — eval 곡선, step / eval_return / eval_length[env step, 20ep 평균])**
```
0   : -2.7 / 592      10k: 21.5 / 138     20k: 21.5 / 139     30k: 26.7 / 160
40k : 27.0 / 160      50k: 30.7 / 186     60k: 83.3 / 385     70k: 72.7 / 320
80k : 99.3 / 410      90k: 95.6 / 396    100k:180.3 / 681    110k: 93.2 / 376
120k:296.2 / 920     130k:224.2 / 669    150k:106.5 / 392    160k:336.7 / 932
```
**(증거 — lap 완주, log_completed=1.0[2랩 완주]은 단 6회, 전부 step 129k 이후)**
```
첫 비영(非零) log_lap_time_s: step 111504 (≈19.5s, 미완주)
첫 2랩 완주(log_completed=1): step 129280 (lap_time 37.6s)
이후 완주: 164976/165904/166688/168320/169728 (lap_time ≈ 35.5~36.2s)
```
- env step = 0.02s. 1랩 ≈ 800~930 step ≈ **16~19초**. eval_length가 138~410(=2.8~8.2초)인
  구간은 **랩을 완주하지 못하고 크래시**한다는 뜻.
- **저장된 스냅샷(≤85k)의 평균 eval_length는 최대 410(8.2초) → 어떤 저장 스냅샷도 평균적으로
  1랩을 완주하지 못한다.** 완주 능력은 step 111k+ 이후에 ~18초대로 등장(스냅샷 미저장 구간).
- step 0(미학습)이 오히려 eval_length 592로 가장 긴 것은 **거의 정지에 가까운 행동 → 느리게
  표류하다 timeout**일 뿐 진행이 없음(eval_return -2.7).

**결론 (Critical):**
1. **"~100초에 *완주*하는 policy"는 이 학습 런에 존재하지 않는다.** 진행은 [크래시(~3초)] →
   [고속 완주(~18초)]로 점프하며, 그 사이 "느리지만 완주" 구간이 없다. RL이 자연히 만드는 게
   아니다(느리고 정밀한 주행은 별도 역량).
2. 따라서 **계획 D2-방법A(초기 체크포인트=100초 policy)는 거짓** — 크래시 데이터만 나온다.
3. **방법B(빠른 policy speed-cap)가 폴백이 아니라 유일하게 타당한 경로**다. 계획은 "dynamics 분포
   변화"를 이유로 B를 비선호했으나, A가 불가능하므로 **B로 전환 + §3의 캡-범위 다양화**가 정답.
   차체 물리값은 불변, action(speed)만 스케일하므로 "차체 물리 무변경" 제약 위반 아님.
4. warm-load 짧은 재학습도 **100초 완주 policy를 만들지 못한다**(같은 크래시→고속 진행을 반복).
   폐기 권장.

**과제 제약 구분(사용자 지시대로):** "환경 추가 상호작용 없이"는 **offline RL 학습 단계**에 대한
제약이다. behavior policy로 데이터를 *수집*하는 것은 허용된다(데이터 생성). speed-cap rollout은
데이터 수집이므로 제약 위반이 아니다. ✅ 이 구분은 계획·검토 모두 동일.

---

## 5. D4 (value 대상 reward) — progress-dense + collision penalty, **lap 보너스는 제외**

npz reward 성분 실측(`log_reward_progress`[0~0.19/step, dense], `log_reward_collision`[-10, 종단
희소], `log_reward_lap`[Oschersleben +100, 극희소], `reward`=합산).

- value = 할인 return-to-go(§1.3). guidance gradient의 품질은 **dense·매끄러운 reward**일수록 좋다.
- **권고: value 학습용 reward = `log_reward_progress` + `log_reward_collision`(축소 가능).**
  - progress가 dense → ∇value가 매 step "더 멀리/빠르게"를 가리킴(lap time과 직접 정렬). ✅
  - collision penalty 유지 → "빠르되 안 박는" 균형. 단 -10은 progress(~0.1)보다 100배 스케일이라
    value 분포를 종단에서 지배·고변동 유발 가능 → **-2~-3 수준으로 축소 또는 normed value 사용**
    (`ValueDataset(normed=True)`가 [-1,1] 정규화 제공, [sequence.py:111-135]).
- **lap 보너스(+100) 제외 권고**: 극희소(완주 시 1회) → value 타깃 분산 폭증, guidance엔 거의
  기여 없이 학습만 불안정. lap time 최적화는 progress dense항이 이미 대리한다. (보고서엔 "full
  reward vs progress-only 비교"를 파일럿으로 남기되, 기본은 progress+소형 collision.)
- ⚠️ 속도캡 데이터의 함정: value는 데이터의 return을 학습한다. ×0.16 캡 데이터만 있으면 "최대
  progress"가 낮게 학습됨 → guidance가 끌 천장도 낮음. §3의 캡-범위 데이터가 value 학습에도 필수.

---

## 6. 완결성·순서 — 빠진 단계와 미검증 가정

**P0~P6 그대로 가면 성공하는가? — 부분적. 아래를 보강해야 "100초 baseline보다 빠른 2랩 완주"에
실제 도달한다.**

**누락/보강 단계:**
- **[신규 P1] "behavior/데이터 policy를 *만드는* 단계"가 명시돼야 한다.** 현 P1은 "초기 체크포인트
  확보"인데 그게 불가능(§4). → "expert(policy_best_lap16.6s_step85k) + 속도캡 범위 {0.16,0.27,
  0.4,0.65,1.0} 정의" + **각 캡의 실제 lap time을 rollout로 측정**(baseline 숫자 확정).
- **[게이트 보강 P1] baseline lap time 실측**: "×0.16이 정말 ~100초에 완주하는가"를 rollout로
  확인(speed scale↔lap time 비선형 가능). 성공 기준의 기준점이므로 측정 없이 진행 금지.
- **[P5 보강] GuidedPolicy↔f1tenth env 인터페이스 3종 명세 필요:**
  - (i) **normalizer 직렬화**: SequenceDataset의 LimitsNormalizer(per-dim min/max)를 학습 시
    저장하고 추론 시 동일하게 로드해야 함([policies.py:33,39] unnormalize 의존). 계획에 없음.
  - (ii) **conditioning**: 매 env step의 현재 obs(=피처/축소-lidar)를 정규화→`{0: obs}`로 넣고
    ([sequence.py:73-77], [helpers.py:142-145] apply_conditioning이 obs부분만 t=0에 고정),
    출력 첫 action을 unnormalize→env action으로. centerline-피처 표현이면 **online 피처계산
    모듈**(closest_idx/arclen→e_y,e_ψ,κ)이 추가로 필요(P3에서 구현).
  - (iii) **MPC 재계획 주기**: 매 step 전체 reverse diffusion(n_diffusion_steps=20 × n_guide=2
    grad) 호출은 비싸다. **K-step open-loop 실행 후 재계획**(예 K=4~8)으로 비용 절감 + 계획에
    명시. action_repeat 2와 곱해짐 유의.
- **[Markov 점검]** Diffuser는 **현재 관측 1프레임만 조건**으로 받는다([helpers.py:142-145];
  history 없음). Dreamer의 RSSM(순환)과 달리 obs가 Markov해야 한다. lidar+[v,yaw,slip,prev_a]는
  근사 Markov, centerline-피처는 더 명확히 Markov → §2의 피처 표현이 Markov 측면에서도 유리.
- **[sampling 지연/sim 평가]** 계획 주장 "plan마다 sim 멈춤 → sim시간=lap time" ✅ 타당. wall-clock은
  100초 랩 × 5000 step × 20 diffusion = 큰 값이나 sim-time 무관. K-step 재계획으로 wall-clock도 완화.
  보고서에 "lap time은 sim-time 기준, 추론 wall-clock 별도 보고" 명시.

**순서 권고(수정):** P0(환경/디커플 smoke) → **P1(expert 확보 + 캡범위 정의 + baseline lap 실측)**
→ P2(캡범위 rollout 수집, raw lidar+state 저장) → P3(표현 결정: centerline-피처 우선 + 로더 +
normalizer 저장) → P4(diffusion+value 학습, 생성 궤적 품질 점검) → P5(GuidedPolicy 연결: normalizer
로드+online피처+K-step MPC) → P6(평가: baseline 대비 lap time, 2랩 완주).

---

## 7. 실현성 (torch/메모리/트랙)

- **torch 1.9→2.4 (py3.8 RL_project venv 재사용)**: 핵심 코드 API 드리프트 사소(§1.4). ✅ 단
  scripts/utils 글루는 미감사 → P0 smoke에 **train.py 1-step 학습까지** 포함해 글루를 조기 노출.
- **8GB GPU**: §1.2대로 binding 아님. centerline-피처(transition_dim~12) 또는 다운샘플(~70)이면
  여유. raw 1080도 메모리상은 가능(생성 난이도가 문제). horizon 32~64, batch 32 권장.
- **Oschersleben 단일(자기교차·고속)**: 자기교차는 lidar 표현에선 모호성 유발 가능(같은 lidar가
  다른 트랙 위치) → **centerline-피처(arclen 기반)가 자기교차 모호성도 해소**(또 하나의 §2 근거).
  고속 구간은 캡-범위 데이터가 커버.

---

## 8. 리스크 Top 3 + 미반영 완화책

| # | 리스크 | 심각도 | 계획 반영 | 완화책(신규) |
|---|--------|--------|-----------|--------------|
| 1 | **100초 완주 behavior policy 부재**(§4) | **Critical** | ❌(전제 오류) | expert speed-cap 범위로 *제작* + baseline lap 실측 게이트 |
| 2 | **데이터 커버리지=향상 천장**(§3) | Major | ◐(원리만, 방법 틀림) | 캡 스펙트럼{0.16~1.0}으로 빠른 재료 포함 |
| 3 | **표현 생성 난이도 + 자기교차 모호성**(§2) | Major | ◐(lidar 다운샘플만) | centerline-피처 저차원 state 1순위 전환 |
| (4) | scripts/utils 글루 미감사·normalizer 직렬화 | Minor~Major | ❌ | P0에 1-step 학습 smoke, P5에 normalizer 저장/로드 명세 |

**과잉/과소 (개인 과제 규모):**
- **과소**: D2 검증(§4)이 계획에서 누락 — 가장 치명적. baseline lap 실측 게이트 필수.
- **과잉**: 없음. 오히려 캡-범위 데이터·centerline 피처는 공수 대비 성공률을 크게 올리는 합리적
  추가. 단 캡 5종은 많으면 3종{0.16,0.4,1.0}으로 축소해도 충분(개인 과제 규모).

---

## 9. 모델 선택 재확인 (가볍게)

코드 분석 결론 "Diffuser가 이 과제에 맞다"를 **흔드는 치명적 미스매치는 없다.** offline-native,
value guidance로 return 직접 최적화, 네이티브 state 사용, 단일 관측 조건(Markov)까지 f1tenth에
부합. ICML 2022 검증성도 사용자 선택 기준과 일치. **Diffuser 유지 타당.**

단 한 가지 환기: Diffuser의 강점은 "데이터 안에서 좋은 걸 재조합·유도"이지 "데이터를 넘어선
외삽"이 아니다. 그래서 **이 프로젝트의 성패는 모델이 아니라 §3·§4의 데이터 설계에 달려 있다.**
(대안 모델 재탐색은 불필요.)

---

## 10. 계획 수정안 diff 요약

```
[배경/D1]
- "lidar 1080을 채널로 → 무겁고 8GB 메모리 부담" 
+ "메모리는 binding 아님; 진짜 리스크는 '미래 lidar 생성 난이도'"
- D1 우선순위: (a)다운샘플 > (b)encoder > (c)centerline피처
+ D1 우선순위: (c)centerline-피처 저차원 state(~10차원) 1순위
+   > (a)다운샘플32~64 또는 (c)+lidar 하이브리드 백업; raw는 npz에 보존해 표현 디커플

[D2 — 전면 수정]
- "초기 체크포인트(step_5k)=100초 policy; 없으면 warm-load 재학습"
+ "[반증됨] 저장 스냅샷 중 1랩 완주하는 것 없음(초기=크래시, 완주는 111k+ ~18초).
+  → expert(policy_best_lap16.6s_step85k)에 speed-cap 범위 {0.16,0.4,1.0} 적용해
+  데이터 생성. ×0.16(≈100초 추정)을 baseline으로 명시하고 lap time을 rollout로 실측."

[향상/천장 §]
- "다양성 = 여러 초기 체크포인트 + action noise"
+ "다양성 = expert 속도캡 스펙트럼(완주 라인 유지). 빠른 재료가 데이터에 있어야 100초↓ 합성 가능."

[D4]
+ "value reward = progress(dense) + collision(-2~-3로 축소) ; lap 보너스(+100) 제외(희소).
+  normed value([-1,1]) 사용 권장."

[Phase]
+ P1: expert 확보 + 캡범위 정의 + 각 캡 baseline lap time 실측(게이트)
+ P3: 표현 결정 + npz 로더(에피소드 단위 yield) + LimitsNormalizer 저장
+ P5: normalizer 로드 + online centerline-피처 계산 + K-step(4~8) MPC 재계획
+ P0: d4rl 디커플 시 sequence.py 상단 import + load_environment/seed stub 포함,
+     train.py 1-step 학습 smoke까지(글루 조기 노출)
```

---

## 11. 확인 못 한 항목 (침묵 금지)

1. **scripts/train.py · train_values.py · plan_guided.py의 글루 코드**(Config 객체, diffuser.utils,
   dataset/model 빌드 배선)를 라인별로 감사하지 못함. 핵심 모델/샘플링/데이터계약/config 기본값은
   검증 완료. 학습 스크립트를 우리 환경에 붙이는 공수가 숨은 리스크(§1.4, §8-#4).
2. **torch 1.9→2.4 실제 구동 smoke 미수행**(P0 대상). API 드리프트는 사소 판단이나 실측 아님.
3. **f110_gym + py3.8 venv에서 GuidedPolicy 추론 동거** 가능성은 [[003-diffuser-code-analysis]]
   §5 주장을 신뢰; 직접 실행 검증 안 함.
4. **×0.16 speed-cap이 정말 ~100초에 *완주*하는지** 미검증(speed↔lap 비선형, 저속에서
   조향/안정성 변할 수 있음). P1 게이트로 반드시 rollout 실측 필요 — 만약 저속캡이 완주를
   못 하면(저속 불안정) 캡 비율 재탐색 또는 progress-reward로 저속 완주 policy 짧게 fine-tune.
5. **centerline-피처의 online 계산 정확도**(closest_idx 점프·자기교차 구간 곡률 테이블) 미구현·미검증.
   P3 구현 시 검산 필요.
6. RL_project `tools.simulate`가 speed-cap action 후처리를 깔끔히 주입할 수 있는지(액션 파이프라인
   훅 위치) 미확인 — P2에서 확인.

— 끝. (다음: 이 검토 반영해 001을 003 plan으로 개정하거나, P0/P1부터 구현 진입.)

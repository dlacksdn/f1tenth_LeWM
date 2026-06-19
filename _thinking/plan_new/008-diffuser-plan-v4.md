# 008 — Diffuser offline RL 확정 계획 v4 (속도상한 훈련 (B) 확정 + 제약·비용·게이트 정밀화)

> 2026-06-18. plan_new/006(v3)을 잇는 **현재 SSOT**. critic v3([[007-diffuser-critique-v3]])를 반영하고,
> 사용자 확정(=(B) 속도상한-훈련 직행, ~11–16h 비용 수용; (A) 탐침 미채택) + 신규 제약(차량/환경 물리
> 무변경, Dreamer 훈련 기능 무침범)을 통합한다. **006 D2는 본 문서로 정밀화·확정.** 모든 lap=2랩.
> append-only(006/007 보존). 선행: [[006-diffuser-plan-v3]], [[007-diffuser-critique-v3]],
> [[005-diffuser-critique-v2]], [[006-glue-correction-and-data-contract]], [[005-diffuser-venv-and-handoff]],
> [[004-dreamer-reuse-and-behavior-policy]], [[003-project-spec]] + PDF.

---

## 배경 (요약 — 상세는 006)
과제(AIE4003 개인, Offline RL): 느린 policy로 데이터 수집 → 환경 추가 상호작용 없이 더 빠른 정책 학습.
산출=behavior policy보다 빠른 2랩 lap time + 보고서. 트랙=Oschersleben 단일(map_easy 무관). 모델=Diffuser
(궤적 diffusion + value guidance, in-distribution 생성기). 평가=정성적.

---

## 006 → 008 변경 요지 (007 반영)

| # | 변경 | 근거 |
|---|------|------|
| Δ1 | **(B) 속도상한-훈련 = 주경로 확정**((A) rollout-clamp 탐침 미채택 — 사용자가 (B) 데이터 품질 선호 + ~11–16h 수용) | 사용자 확정 |
| Δ2 | **Stretch(expert 초과) 폐기 → Floor = cap-15 초과로 확정** | 007 §2: 속도상한 정책은 expert보다 전 구간 느림 → sub-16.6s 재료 부재 → 초과 불가 |
| Δ3 | **신규 제약 섹션**: 차량/환경 물리 무변경 + Dreamer 훈련 무침범(V_MAX 기본20 파라미터화) | 사용자 지시 + 007 §2-#3(V_MAX 하드코딩) |
| Δ4 | **(B) 비용 정량화 + wall-clock 게이트**(~5.5–8h/캡, 조기종료) | 007 §2-(B-c), #4 |
| Δ5 | **데이터량·다양성 게이트**(tier당 ≥N ep, rollout 시 action noise) | 007 §2(완주 6ep), 데이터 설계 |

---

## ★ 핵심 제약 (절대 준수)
1. **차량/환경 자체 파라미터 무변경**: 차량 동역학(질량·마찰·휠베이스 등)·env reward·종료조건·맵을 일절
   수정하지 않는다. **우리가 바꾸는 것은 오직 action space의 속도 상한 V_MAX 하나.**
2. **Dreamer 훈련 기능 무침범**: V_MAX를 **기본값 20인 설정 파라미터(`__init__` 인자 또는 action wrapper)로
   승격**한다. Dreamer의 기존 호출(인자 없음)은 V_MAX=20 그대로 동작 → **Dreamer는 언제든 정상 훈련 가능.**
   캡 런만 V_MAX=10/15를 전달. (`f1tenth_env.py:36`의 하드코딩 상수 *직접 손편집 금지* — 백워드 호환 파라미터화.)
   - 구현 기제(init-arg vs wrapper)는 P1 디테일. 단 **NormalizeActions가 action_space.low/high를 동적 매핑**
     ([wrappers.py:35-39])하므로 상한만 바꾸면 명령 매핑은 자동 반영.
3. **reward 무변경**: 캡 정책 훈련은 **Dreamer의 기존 reward 그대로** 사용(속도만 제한). Diffuser value용
   D4 reward(progress+축소 collision, lap 제외)는 *수집 npz의 로그 성분에서 Diffuser 학습 시 별도 구성*.
4. **`_STATE_SCALE=20` 유지**([f1tenth_env.py:50]): 캡 데이터 속도 피처가 하단 범위에 몰리나 warm WM이 그
   스케일로 학습됐으니 불변(007: 양성 추정).

---

## 설계 결정

### D1. 관측 표현 — lidar 다운샘플(~64)+state, pose는 rollout때 기록만 *(006 유지)*
v1=lidar로 파이프라인 관통, centerline 피처는 v2 옵션(pose는 P2에서 함께 기록).

### D2. ★ 데이터소스 — 속도상한 훈련 정책들 + 기존 best expert (확정)
- **반증 화해(007 ✅ 검증)**: 005가 반증한 건 *무제한* warm-load 재학습(고속 점프). **속도상한 훈련은
  V_MAX를 못 넘어 고속 점프가 불가능 → 그 속도대 완주로 수렴.** 미검증≠반증, 1차 소스로 지지됨.
- **느림·중간 tier = warm-load(stage1_map_easy3 WM, actor-critic fresh) → Oschersleben에서 직접 훈련**:
  - **V_MAX=15 → cap-15 정책**(중간; **먼저 훈련** — expert 속도대에 가까워 완주 학습 가능성↑)
  - **V_MAX=10 → cap-10 정책**(느림, baseline)
  - 각 정책이 그 속도에 맞는 조향을 *학습* → 깨끗한 완주 데이터.
- **expert tier = 기존 best `policy_best_lap16.6s_step85k`(2랩 ~33s) 그대로**(재훈련 0).
- **비율 잠정 45(cap-10)/45(cap-15)/10(expert)**, P4 결과 보고 조정(007: expert 10%는 expert-light, P4 점검).
- 예상 2랩(007 실측 보강, P1 확정): cap-10 ~60–68s / cap-15 ~44–46s / expert ~33–38s. "60→30" 서사 정합.
- **교수님 expert 허용 가정**. 불허 시 expert tier 제거(캡-10/15만; 천장 cap-15) — **비블로킹**(캡 훈련은 답과 무관).

### D3. 속도 제한 = *훈련 시* action space V_MAX (위 제약 2)
rollout-clamp (A) 미사용(사용자가 (B) 품질 선호). V_MAX 파라미터화로 적용, 차체물리 무변경.

### D4. value 대상 reward *(006 유지)*
progress(dense)+축소 collision(−2~−3)+lap 보너스 제외+normed value([-1,1]). Dreamer reward와 별개로 Diffuser 학습 시 구성.

### D5. Diffuser 글루 *(006 §D5 / [[006-glue-correction-and-data-contract]] §2 그대로 — 변경 없음)*
rendering(imageio/mujoco/video/d4rl)·colab·preprocessing·sequence guard + buffer.py:12 np.int64 + f1tenth 로더
+ config(clip_denoised=True, termination_penalty=None) + normalizer 별도 저장(Trainer.save 미저장) + 피처 parity.

---

## 성능 기대치 / 성공 기준 (007 반영)
| 수준 | 내용 | 비고 |
|---|---|---|
| **Floor(성공)** | **cap-15 정책보다 빠른** 2랩 완주 | value guidance가 expert(10%) 재료로 cap-15 위로 끌어올림. 달성 목표 |
| **Realistic** | best(~33s) 수준 근접 | "느린 데이터에서 expert급 복원" — 보고서로 당당 |
| ~~Stretch~~ | ~~expert 초과~~ → **폐기** | 속도상한 정책은 전 구간 expert보다 느림 → sub-16.6s 재료 부재 → 본 데이터 구성상 불가(007). 노리면 v2(라인 다양성 별 정책) |

---

## 데이터-모델 궁합 (★ 보고서 필수 서술 — 사용자 지정, 006 유지)
> 데이터는 "느림(저속상한 훈련)+빠름(expert)"의 **이중분포(bimodal/medium-expert)**. 단순 BC는 두 모드를
> 평균 내 망가지나, **value 기반·궤적 생성(Diffuser)은 이 혼합을 다루도록 설계**(value guidance로 고-return
> 선택, stitching으로 좋은 구간 재조합). unimodal보다 어려운 세팅이라 **Diffuser 선택이 적절.** (D4RL
> medium-expert/medium-replay가 이 표준 벤치마크.)

---

## Phase 분해 (게이트 + 런타임 배치)

| # | 작업 | venv | 게이트 |
|---|------|------|--------|
| **P0** *(훈련과 병행 가능)* | 글루 디커플(D5) + 더미 로더 train.py 1-step | **새 .venv** | (A)TemporalUnet import (B)f1tenth 로더로 SequenceDataset 적재 (C)train.py 1-step loss (D)train_values/plan_guided 파서. Config pickle→Renderer 확인 |
| **P1-impl** | **V_MAX를 기본20 파라미터로 승격**(init-arg/wrapper) + cap-10/15 비손편집 전환 | RL_project venv | Dreamer 무인자 호출 V_MAX=20 정상 + 캡 런 10/15 전달 확인 |
| **P1-train** | **warm-load → Oschersleben V_MAX=15·10 정책 훈련**(cap-15 먼저) | RL_project venv | 각 캡 **2랩 안정 완주** + 2랩 시간 실측. **wall-clock 상한 ~8h/캡, 완주 안정화 시 조기종료**. baseline=cap-15 시간 정의. 미달 시 캡/하이퍼 조정 |
| **P2** | cap-10·15 정책 + best rollout → **lidar+raw+pose 기록**, **action noise로 다양화**, tier당 **≥N ep(예 ≥30, 가능하면 수백)** 수집(크래시 폐기) | RL_project venv | tier별 완주율·최소 ep 수·궤적 다양성 리포트 |
| **P3** | f1tenth 로더 + lidar 다운샘플 + normalizer 라운드트립 검산 | 새 .venv | SequenceDataset 통과 + 동일 bounds |
| **P4** | diffusion+value 학습 + **생성 궤적 품질·천장 점검**(expert 10%로 value가 고-return 선택하는지) | 새 .venv | loss 수렴 + 품질. expert 비율 부족 시 상향 |
| **P5** | plan_f1tenth.py: GuidedPolicy↔f110_gym, normalizer 로드, K-step(4~8) MPC + wall-clock 추정 | 새 .venv | 1 ep 완주 시도 |
| **P6** | 평가: cap-15 대비 2랩 lap time, 2랩 완주, best 근접 | 새 .venv | **cap-15보다 빠름(floor)** |

---

## 리스크 Top + 완화
1. **캡 정책이 2랩 완주 학습 실패/지연** → P1 게이트(완주+wall-clock 8h 상한+조기종료). 저속은 완주 쉬움(WM이 ≤10대 풍부히 학습, 007: 크래시 ep 평균 10.6m/s). 안 되면 캡/리워드 조정.
2. **데이터 빈약/다양성 부족** → P2 tier당 ≥N ep + action noise. Diffuser는 데이터량 선호.
3. **expert 10% 신호 부족** → P4 점검 후 비율 상향(잠정).
4. **V_MAX 파라미터화가 Dreamer/normalize 경로를 건드림** → 기본20 백워드호환, Dreamer 무인자 호출 정상 확인 게이트(P1-impl).
5. **글루 디커플 잔여** → P0 확장 게이트.
6. **normalizer/피처 seam** → normalizer 라운드트립(P3), 피처 단일함수(v2).

---

## 다음 단계 (인수인계)
1. **지금 = P1-impl + P1-train 착수**(밤샘 가능): V_MAX 기본20 파라미터화 → **cap-15 먼저 훈련**(Oschersleben, warm-load), 2랩 완주·시간 확인. 이어 cap-10.
2. **P0 병행**(새 .venv): 글루 디커플 + 1-step + 파서. 훈련과 독립.
3. **교수님 확인**(비블로킹): expert(기존 best) 정성평가 포함 가능? 불허 시 expert tier 제거.
4. 분기마다 _thinking 문서 + commit + push(상시 규약, 자율 pull 금지). f1tenth 판단 시 [[003-project-spec]] + PDF.

---

## ★ 검수 반영 + 성공기준 재정의 (2026-06-19 밤, append — 적대적 critic 검수 후)

> 적대적 critic(새 세션) 보고 + 내 1차소스 재검증(crash_data 직접 로드)으로 위 성공기준을 정정한다.
> **008 본문(위)은 보존**, 본 절이 성공기준·reward·평가의 최신 SSOT. 아키텍처 이해는 [[understand/001-diffuser-vs-value-architecture]].

### 검수 판정: critic (C)"설계상 실패" → 재검증 후 **(B)**
- critic 주장(검증됨): 데이터 내 최속 *2랩 완주* = cap-10 **56.14s**, 37.3s보다 빠른 완주 **0/52**.
  → 구 floor(cap-15 37.3s 초과)는 **2랩 완주 라벨이 데이터에 없는 영역**.
- 내 재검증(반박 근거, 직접 실측 crash_data 767ep): **고속 *단일랩* 재료는 빽빽** — cap-20 117랩@18.04s,
  cap-15 132랩@19.74s(2랩 환산 ~36–40s). 2랩 완주 0은 *고속 불가*가 아니라 stochastic 정책이 2랩 도는 중
  충돌하기 때문. **obs는 lap-blind + 평가는 K=1 MPC**(매 step 재계획, policies.py:36 `action[0,0]`)라 모델은
  2.56s 윈도만 이어붙이고 그 고속 윈도는 in-distribution. → floor는 *OOD 불가능*이 아니라 **불확실 stretch**.

### ★ 성공기준 재정의 (위 "성능 기대치" 표 대체)
| 수준 | 내용 | 데이터 지지 |
|---|---|---|
| **승리(확정 목표)** | **baseline(cap-5 107.16s) 초과** | cap-10 완주 56s가 이미 2× → BC로도 자명 |
| **목표(realistic)** | **≈56s**(cap-10 완주 수준 복원) | 완주 52개 in-dist |
| **stretch** | **≈36–40s**(stitching+value-shaping 베팅) | 고속 단일랩 249개 존재하나 충돌과 얽힘 → 불확실 |
| ~~구 floor~~ | ~~cap-15 37.3s 초과~~ | **stretch로 강등**(2랩 완주 라벨 부재) |

보고서 서사: "expert급 복원" ❌ → **"충돌-위주 데이터에서 안전 완주 복원 + stitching 탐색"**.

### ★ 진짜 linchpin = value 설계 (critic D3)
value는 speed를 보상(corr(speed,R)=+0.36)하고 **충돌은 γ=0.99로 소멸**(collision −10·lap +200 모두 discounted)
→ "안전-고속"과 "충돌-고속"을 구분 못 함. stretch의 실제 리스크이자 **해결 레버**.
- **v1 = 현 학습 유지**: diffusion + **npz 결합 reward** value([f1tenth.py:117]; npz에 log_reward_{progress,collision,
  reverse,diverged,lap} 성분 분리 존재).
- **D4 정정**: 위 D4의 "축소 collision −2~−3"은 **D3를 악화**(페널티↓ = value가 충돌 더 못 봄) → **npz −10 유지가
  정답**. reward 재구성 안 함, 문서만 정정.
- **D3 contingency**: v1 평가(P5/P6)서 value→충돌 징후 나오면 **value만 ~19h v2 재학습**(crash-aware:
  collision 성분 강조 또는 짧은 discount). diffusion 재학습 불필요. (over-plan 금지 — 징후 확인 후.)

### 기타 검수 항목
- **D5(실버그) 수정완료**: `config/f1tenth.py` `plan.discount 0.997→0.99`. value는 d0.99로 저장되어
  value_loadpath의 `{discount}`가 0.99여야 P5 eval 로드 성공. plan 블록=eval 전용이라 **학습 무영향**.
- **S3**: 평가 **K=1**(매 step 재계획) 확정 — open-loop drift 없음, sim이라 latency 무관(lap=sim-time).
- **S4**: P5에서 raw action → env `[-1,1]` 재정규화는 **V_MAX=20 고정**(per-tier 값 금지).
- **S5**: 위 "코어 무변경"엔 ValueFunction fork-patch 1건 예외(원본 off-by-one 버그, 무회귀 검증).
- **S1**(패딩 14%→정지 편향, min ep143≥H128이라 v2서 use_padding=False로 제거 가능)·**D2**(medium-expert
  서사 정정)은 v2/문서 단계로 보류.

### 측정 가능성 (P5 게이트 전제)
discount 일치(D5 ✅) + normalizer 통계 pickle 고정 + raw↔`[-1,1]` V_MAX=20 왕복 + plan_f1tenth.py 구축
후에야 floor/baseline 측정 가능. (현재 P4 학습 중, P5 미구현.)

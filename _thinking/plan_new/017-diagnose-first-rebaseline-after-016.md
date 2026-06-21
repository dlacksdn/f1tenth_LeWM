# 017 — 016 검수 수용: "진단 먼저, 베팅 나중" 재배열 + Phase 0 착수

> 2026-06-21. [[016-adversarial-critic-of-015-diversification]](판정 **조건부 채택**)를 전면 수용해
> 015 계획을 재배열한다. **핵심 전환: covariate에 GPU 수~수십시간을 베팅하기 전에, 어느 병목인지
> 싸게 먼저 가른다.** append-only. 엄밀(파일·라인·수치) + 쉽게(표·비유). 본 문서 시점에 Phase 0 착수.

---

## 0. 한 줄
cap10·cap5 둘 다 BC 완주 0/10이지만 **실패 양상이 다르다** — cap10은 66 step(1.32초)에 **즉발 붕괴**,
cap5는 203~1064 step에 걸쳐 **점진 실패**. 015는 이 둘을 "복구 covariate 부재" 하나로 묶었는데, 016이
**서로 다른 두 메커니즘**(cap10=質/obs 냄새, cap5=covariate 정합)임을 지적했다. 게다가 015의 처방
(margin0 cap10에 노이즈 주입)은 **조향 포화 정책이라 복구를 시연할 권한이 없어** 작동 불가. → **결론2를
작업가설로 강등하고, 노이즈 주입을 폐기하고, env var 한 줄짜리 obs 프로브로 갈래부터 가른다.**

---

## 1. 016이 바꾼 것 (수용 목록)
| 016 판정 | 수용 후 변화 |
|---|---|
| **결론2(복구 covariate 병목) = 과잉해석(D1)** | "입증된 병목" → **미입증 작업가설**로 강등. "둘 다 0/10"은 covariate·obs·質을 **구분 못 함**(under-determination) |
| **노이즈 주입 폐기(D4·NR1)** | cap10은 7.76% step에서 조향을 **물리한계(±0.4189 hard clip, f1tenth_env.py:35·322)** 에 명령 → 포화 구간에 노이즈 줘도 복구 시연 **권한 0**. → 노이즈 주입 폐기 |
| **즉발 vs 점진 두 모드(NR2·NR3)** | cap10 BC len[47~96] ↔ cap5 BC len[203~1064] **완전 비중첩**. cap10=즉발붕괴(質/obs), cap5=점진(covariate). **다른 원인** |
| **obs 프로브 선행(D7·NR5)** | `F1TENTH_LIDAR_DOWNSAMPLE=256` flip 1판이 한 갈래를 가르는데 Phase4로 미룬 건 정보경제 역순. **진단을 Phase 0로 선행** |
| **시작점 랜덤화 "바로 가능" 거짓(D5)** | 수집경로 어댑터가 `reset()` 무인자(`collect_crash_data.py:128/224`→어댑터 `f1tenth.py:126-128`)라 통로 끊김. 015가 든 `measure_gap_follower.py:63`은 **다른 경로**. → 어댑터 `reset(options)` 글루 선행 필요(저비용 아님) |
| **"5.5배 버틴다" ~2배 과장(D2)** | 거리환산 2.61배(저속이라 같은 거리를 더 많은 step으로). "마진은 생존을 ~2.6배 늘리나 완주엔 불충분"으로 정정 |
| **NR5: 기저 산출물 베팅 격상** | 012가 BC 완주 대전제를 0/10 반증 → 다양화 fix는 "stretch 보너스"가 아니라 **보고서 기저(baseline 초과 완주) 전체가 걸린 高위험 베팅** → 저비용 갈래가르기 가치 결정적 |

> 016이 확인해준 것(015에 유리): **수치는 2회 교차 재계산으로 전부 정확**(8개 품질지표 + V1), 그리고
> **train-eval 하네스 정합**(수집·평가 동일 `make_env`+`reset()`, obs/action 왕복 보존) → 0/10은
> 하네스 버그가 아니라 **진짜 닫힌루프 현상**. 결론1(마진≠완주)도 견고.

---

## 2. 재정의된 문제 — 세 갈래 (under-determination)
"BC가 완주 못 함"의 원인 후보가 **세 개**이고, 현 데이터는 이를 구분 못 한다:

| 갈래 | 뜻 | 냄새 나는 증거 |
|---|---|---|
| **質 (margin0 한계주행)** | cap10이 코너 무감속·조향 포화라 오차 흡수 여유 0 | cap10 66 step **즉발붕괴**(표류할 시간도 없음, NR2) |
| **obs (인지 부족)** | lidar128 공간해상도/절대 heading 부족으로 코너 판단 실패 | 둘 다 실패. ★단 동역학(속도·yaw-rate)은 state5에 **있음**(NR4) → 살아남는 obs가설=공간해상도·heading |
| **covariate (복구 데이터 부재)** | 라인 밖 표류 시 복구 데모가 데이터에 0 | cap5 **점진 실패+롱테일**(NR3) |

→ **목표: 가장 싼 실험부터 돌려 이 셋을 가른다.**

---

## 3. 재배열 계획 (진단 → 갈래별 처방 → value)

### Phase 0 — 진단: 어느 갈래인가 (저비용 선행) ★지금 착수
| 실험 | 무엇 | 가르는 것 | 비용 |
|---|---|---|---|
| **0a. 폐루프 덤프** | cap10 BC 재실행 + per-step (cmd_v·cmd_steer·정면 lidar) 덤프 → 거동·발산 시점 분석 | **즉발**(처음부터 위태=質/obs) vs **점진**(정상 추종 후 이탈=covariate) | 평가 1판 + 덤프 글루(제일 쌈) |
| **0b. obs 프로브** | `F1TENTH_LIDAR_DOWNSAMPLE=256` + cap10 30ep 재학습 → BC(K1/K3) | obs 공간해상도 병목? (완주·생존↑=obs / 불변=배제) | 재학습 1판 ~1-2h, env var만(코어·어댑터 무변경) |

### Phase 1 — 갈래별 처방 (Phase 0 결과로 분기)
- **obs면** → 고해상도 lidar(256+)로 해결 (수집 불요, 최저비용)
- **質(margin0)면** → **마진 있는 정책**(중간 캡)의 자연 완주 데이터 (cap10은 복구 시연 권한 없음=NR1)
- **covariate면** → **DAgger**(cap10이 표류상태를 *라벨*, 노이즈 불요) 또는 중간캡 시작점-랜덤화 자연수집.
  **노이즈 주입 폐기.** (어댑터 `reset(options)` 글루 선행, 사용자 승인)

### Phase 2 — value(γ0.999) stretch (BC 완주 토대 확보 *후*)
- 011 §8.2 순서 유지(value는 prior가 만든 plan 중 고르기). prior 재학습과는 **순차**(GPU 7/8GB). "완주>충돌"은 γ민감(016 §3.2)이라 약하게만 인용.

### 보고서 프레이밍 (무비용, 지금부터)
- 성공기준 **분리 고정**: (가) baseline 107s 초과 완주 = **모방**(Phase1 산출) / (나) <56s = **stitching**(Phase2, 확률 낮음).
- 다양화 수집은 "covariate-aware 수집(DART/DAgger 계열)"로 정직히 프레이밍. 결론2는 "작업가설".

---

## 4. Phase 0 실행 세부 (본 문서 시점 착수)
- **0a**: `plan_f1tenth.py`에 `--dump_traj`(평가 전용 글루, 코어 무변경) 추가 → step별 cmd_v/cmd_steer/
  정면 lidar(±30°, 인덱스 420:660 min) 기록 → cap10 BC K=1 1판 → 데이터(cap10 완주)의 초반 구간과 대조.
  판정: 처음부터 풀스로틀 급가속+위태 = 質/obs / 정상 추종 후 이탈 = covariate.
- **0b**: `F1TENTH_LIDAR_DOWNSAMPLE=256`로 cap10 prior 재학습(촘촘 ckpt) → 같은 env var로 BC 평가.
  cap10(128) 대비 완주·생존 step 변화로 obs 공간해상도 갈래 판정.

---

## 5. 다음 critic 검수 포인트 (Phase 0 후 재검수용)
- **E1**: 0a의 즉발/점진 판정 기준이 자의적이지 않나? (몇 step·어떤 임계를 "즉발"로?)
- **E2**: 0b에서 256이 완주를 못 내도 "obs 배제"가 정당한가, 아니면 512·heading 추가까지 봐야 하나?
- **E3**: 세 갈래가 **독립 단일**이라는 가정이 맞나? (質+covariate 복합이면 한 갈래 fix로 부분 완주만)
- **E4**: DAgger 경로의 라벨러(cap10 정책)가 표류 상태에서 합리적 복구를 내나? (margin0이면 라벨도 한계)

---

## 6. 함정·규약·참조
- GPU=run_in_background, kill 단독, 학습·평가 순차, 정지 전 state_N>0 확인([[verify-before-kill]]).
- 평가=RL_project .venv·cwd=vendor/diffuser·V_MAX20·2랩, `F1TENTH_MODE=<학습모드>`(normalizer 정합),
  ★0b는 학습·평가 **둘 다 `F1TENTH_LIDAR_DOWNSAMPLE=256`** 으로 맞춰야 obs dim 정합.
- 출력 `--out`/로그 절대경로, `run_logs/` 보존, 촘촘 체크포인트(save_freq2000/n_saves20).
- 코어 무변경(`--dump_traj`·env var는 글루).
- 참조: 검수 [[016-adversarial-critic-of-015-diversification]] / 계획 [[015-overnight-validation-covariate-bottleneck-and-diversification-plan]] / SSOT [[011-staged-gate-plan-v2-finalized]]
- 코드: `scripts/plan_f1tenth.py`(obs구성 54-61, run_episode 64-99) · 로더 `f1tenth.py`(downsample 30·79-94) · 환경 `f1tenth_env.py`(state5 50, clip 322)
- 로그: 신규 `run_logs/{cap10_bc_dump, cap10_256_*}.{json,log}`

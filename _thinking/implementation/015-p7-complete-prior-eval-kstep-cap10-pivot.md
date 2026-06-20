# 015 — P7: complete prior 평가(BC·K스윕) + cap10-only 전환

> 2026-06-21. [[011-staged-gate-plan-v2-finalized]] 단계 게이트 v2의 **실행·진단 기록**.
> Track A(완주 prior) 학습 → BC/K-step 평가 → 진단 → cap10-only 전환 결정까지. 011이 계획 SSOT,
> 이 문서는 그 실행 결과. append-only. 다음 세션 인수인계 + 사용자 검토용(엄밀 + 쉽게).

---

## 0. 한 줄
complete prior(완주 52, cap10 가중)로 **P6의 후진/스핀 degenerate는 제거(✅)**했으나 **완주 0/40**(K 스윕
전부 실패). **K=3가 sweet spot**(충돌까지 2배 버팀 = compounding 완화)이나 완주엔 부족 → **prior가 병목**
(cap5 혼합의 보수적 속도) → **cap10-only 재학습으로 mixture 제거**(진행 중). 별도로 **B0 선검증으로 Track B
value 레버 = γ=0.999 확정**(reward 조정은 무효).

---

## 1. 용어 (이 문서에서 새로 쓰는 것)
| 용어 | 뜻 |
|---|---|
| **compounding error** | 닫힌 루프에서 매 step의 작은 예측 오차가 **누적**돼 데이터 밖으로 발산. K=1(매 step 재계획)일 때 특히 심함. |
| **K-step MPC** | 그린 plan의 **앞 K개 action을 재계획 없이 순차 실행**한 뒤 다시 계획. K↑ = open-loop 구간을 늘려 compounding 완화(단 너무 크면 "눈 감고 달리는" open-loop blind). |
| **mixture-averaging** | 출발 관측이 같으면(정지) diffusion이 cap5(느림)·cap10(빠름) **두 모드의 평균**을 생성 → 흐릿하고 보수적인 주행. |

---

## 2. 한 일 (P6 → P7)
011 단계 게이트 v2의 **Track A**를 실행했다:
- **A1 로더**: `datasets/f1tenth.py`에 `F1TENTH_MODE`(all/complete/cap10) + `F1TENTH_CAP10_WEIGHT` 추가
  (코어 `sequence.py` 무변경, code-reviewer APPROVE). complete=완주 52, cap10 가중 W=3(cap10_full 3배).
- **A2 학습**: complete prior diffusion을 **분리 경로**(`diffusion/f1tenth_complete_H128_T20`)에 학습.
  loss가 **step 16k에서 0.003 saturate**(이후 30k까지 0.003 동일 = 더 학습 무의미)라 16k에서 중단.
- **★ 평가 인프라 함정 2개를 선제 차단**(P6 반복 방지, `plan_f1tenth.py` 수정):
  1. **경로 하드코딩** — `diff_lp`가 원본 P6 prior를 가리킴 → `--diff_subpath` 인자 추가(기본=P6, override 가능).
  2. **normalizer 정합** — complete prior는 **complete-stats**로 학습됐는데 평가가 all-stats로 재fit하면
     좌표계 불일치로 망가짐 → **`F1TENTH_MODE=complete`로 평가 실행** 시 `load_diffusion`이 dataset을
     재생성(serialization.py)하며 normalizer를 complete-stats로 재fit → 정합. (1차 확인: 로그에 `52 episodes`.)
- **A3 평가**: BC(scale=0) + **K-step** (compounding 완화용으로 `--K` 추가).

## 3. 결과 (1차 데이터 — 전부 `run_logs/`에 영구 보존)

### 3.1 A3 BC (scale=0, K=1, 10 ep)
- **완주 0/10**, 충돌 길이 median ~174 step(≈3.5초, 트랙 초반).
- **cmd_speed_mean = 5.1~6.0 (10/10 전부 양수)**.
- ✅ **후진 degenerate 완전 제거**: P6는 cmd_v **−4.0**(후진/스핀)이었음 → +5.5. "깨끗한 prior면 후진이
  사라진다"(014 §5-a) 실증. **과적합도 아님**(데이터 memorize가 아니라 전진 주행 생성 = 일반화).

### 3.2 K 스윕 (BC, 10 ep씩)
| K | 완주 | 충돌 len median | len max | cmd_v mean |
|---|---|---|---|---|
| 1 | 0/10 | 174 | 411 | 5.57 |
| 2 | 0/10 | 302 | 598 | 5.19 |
| **3** | 0/10 | **361** | **716** | 4.99 |
| 5 | 0/10 | **177** ↓ | 233 ↓ | 5.11 |

- **K=3 sweet spot**: K를 1→2→3으로 키우면 충돌까지 **2배 오래 버팀**(174→361 step). → **compounding
  error가 실재했고 K로 완화됨**(014 §5-b 확증).
- **K=5는 open-loop blind**: 급락(361→177). "K가 너무 크면 plan만 믿고 달려 충돌"이 실측됨.
- **그러나 모든 K에서 완주 0** → K(compounding)만으론 완주 불가.

## 4. 진단
- ✅ 후진 제거(prior 청소) + ✅ compounding 완화(K=3) = **P6(전 설정 충돌) 대비 분명한 진전**.
- ❌ 완주 0/40 = **prior 자체가 병목**. 근거:
  - 모든 K에서 **cmd_v ~5 m/s로 보수적** = cap5(115s) + cap10(56s) 혼합의 **mixture-averaging**(010 §2).
    cap10 완주를 깨끗이 재현 못 하고 둘의 평균(흐릿/느린 plan)을 생성.
  - 완주 prior가 **52ep로 얇아** 코너에서 covariate shift.
  - **value는 prior가 만든 것 중에서만 고른다**(010 §4) → 완주 못 하는 prior에 고속랩·value를 얹어봐야 또 충돌.

## 5. B0 선검증 — Track B value 레버 확정 (상세 [[011-staged-gate-plan-v2-finalized]] §9)
학습 대기 중 무비용 CPU로 npz RTG(value 타깃)를 γ·reward별 계산:
- **γ=0.99(현재)**: value가 충돌고속(RTG 31) > 안전완주(20)를 선호 → P6 guidance 실패의 정량 근거.
- **γ=0.999**: 역전(안전완주 217 > 충돌고속 152 > 충돌임박 9) + 빠른완주/느린완주 변별 보존.
- **reward 조정(완주 보너스↑·충돌 페널티↑)은 무효**(거리 문제라 크기 키워도 γ로 소멸; 009 B1 폐기).
- → **Track B value = γ=0.999 단독**(B1·reward 조정 폐기).

## 6. 결정 — cap10-only 전환 (토대 → stitching)
- prior가 완주조차 못 하니, **cap5 혼합을 제거** = `cap10_full` 완주 **30ep만**으로 재학습(순수 56s 모드).
- ★ **방향 명시(사용자 확인)**: cap10-only는 **"stitching 개선"이 아니라 "완주하는 토대" 만들기**다(cap10
  정책 모방 = 56s 복원, 출발선). **반드시 Track B**(고속랩 추가 + γ=0.999 value guidance)**로 이어가야**
  진짜 방향(stitching으로 56s 초과)에 도달한다. 순서 = 토대 → stitching. 여기서 멈추면 방향 어긋남.
- **진단 겸**: cap10-only가 완주를 내면 → 원인은 mixture. 안 내면 → 데이터 얇음 → 고품질 수집(cap12 등).
- 학습: `F1TENTH_MODE=cap10` + 분리경로 `diffusion/f1tenth_cap10` + 40k(30ep라 빨리 saturate). 이후 K=3 평가.

## 7. 코드 변경 (코어 무변경; ValueFunction fork-patch 외 예외 없음)
- `datasets/f1tenth.py`: `F1TENTH_MODE`에 **cap10** 추가(`cap10_full` 완주만), `_auto_max_n_episodes` 반영.
- `scripts/plan_f1tenth.py`: **`--diff_subpath/--val_subpath`**(로드 경로 override) + **`--K`**(K-step MPC =
  `traj.actions[0]`의 앞 K개 순차 실행). GuidedPolicy/코어 무변경(`policies.py`가 이미 full plan 반환).

## 8. 함정 / 규약
- ★ **평가는 `F1TENTH_MODE=<학습한 모드>`로 실행**해야 normalizer 정합(complete→complete-stats,
  cap10→cap10-stats; `check_compatibility`는 stats 미검사라 사람이 보장).
- ★ **로그는 `run_logs/`에 영구 저장·삭제 금지**(보고서 그래프 재료). [[run-logs-preservation]]
- GPU 실행 = `run_in_background`(foreground+CUDA=exit144). **잔존 프로세스 kill 같은 복합 셸 로직도 144 유발**
  → 평가/학습은 **단순 단일 명령**으로 발사. 종료 = PID/TaskStop.
- **폐기 없음**: 원본 P6 ckpt + complete ckpt 백업(`logs/f1tenth/_archive_p6/`) + 분리 경로 = 원본 무변경.
- 산출물: `run_logs/{trackA_complete_diff.log(loss), a3_bc10.log, a3_K{2,3,5}.log, a3_eval_16k.log, train_cap10.log}`.

## 9. 다음 계획
1. **cap10 prior 학습 완료 → K=3 BC 평가**(`F1TENTH_MODE=cap10 --diff_subpath diffusion/f1tenth_cap10_H128_T20 --K 3 --scale 0`).
2. **완주 나오면** → Track B: `driving` 모드(완주 + 추출 고속랩 249) prior + **γ=0.999 value** 학습 → scale 스윕 = **stitching/개선**.
   **안 나오면** → **고품질 데이터 수집**(cap-12/13 완주 rollout, RL_project).

## 10. 참조
- 계획 SSOT: [[011-staged-gate-plan-v2-finalized]] (단계 게이트 v2 + §8 γ/데이터 + §9 B0)
- P6 진단: [[013-p6-failure-diagnosis-and-driving-prior-plan]] / 검수: [[014-adversarial-critic-of-driving-prior-plan]]
- 데이터 현황·문제: plan_new/009 / 게이트 계획 검수: plan_new/010
- 로그 보존: [[run-logs-preservation]]

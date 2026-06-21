# 018 — Phase 0 진단 종합 + 마진 데이터 처방(v_max 낮춘 rollout) 계획

> 2026-06-21. [[017-diagnose-first-rebaseline-after-016]]의 Phase 0(진단) 실행을 마치고, 그 결과로
> 내린 처방과 실행 순서를 정리한다. append-only. 엄밀(수치·파일·라인) + 쉽게(표·비유). 본 문서
> 시점에 Phase 1(cap8 타진) 착수.

---

## 0. 한 줄
Phase 0 진단 결과 **완주 실패는 단일 원인이 아니라 복합**이다 — cap10은 **質(margin0 한계주행)** 으로
첫 코너 즉발, cap5는 **covariate(복구 데이터 부재)** 로 점진 실패, obs(lidar 256)는 생존을 1.5배
늘리는 **보조**. 목표(baseline 107s 초과 = cap10 속도대 완주)엔 **"마진 있으면서 baseline보다 빠른"**
데이터가 필요하다. → **cap10 정책의 속도캡만 8로 낮춰 rollout하면 새 정책 학습 없이 그 데이터를 얻는다**
(예상 ~71s/2랩, baseline 107s 초과). 가장 싼 타진부터 돌려 "마진이 완주를 여나"를 가른다.

---

## 1. Phase 0 진단 종합 (확정 — 전부 실측)
| 데이터/축 | 벽 | 실패 양상 | 1차 근거 |
|---|---|---|---|
| **cap10** (56s, 고속) | **質 (margin0)** | 90 step 데이터 추종 후 **첫 코너 즉발** | 0a 덤프: cmd_v·front 데이터와 일치하다 코너서 front 급감+조향 ±0.419 포화 |
| **cap5** (115s, 저속) | **covariate** | 마진 크나 **점진 실패+롱테일**(203~1064 step) | 야간 cap5-BC: 5.5배 버티나 0/10; 0a상 cap10과 실패모드 비중첩 |
| **obs** (lidar 128→256) | **보조** | 생존 1.4~1.8배↑(66→118, 90→127)이나 **완주 0** | 0b: 256 BC median K1=127·K3=118 |

→ **016 E3(복합 원인) 적중.** obs는 생존을 늘리나 완주의 근본 벽은 質. **covariate는 cap10에선
약화**(표류 전 즉발)이나 cap5에선 실재. 즉 **속도대마다 다른 벽**이 작동한다.

### 1.1 핵심 통찰 — 속도-마진 트레이드오프
- cap10(빠름, 마진0) ↔ cap5(느림, 마진 큼). **둘 다 완주 0이나 이유가 정반대.**
- 목표는 **그 사이**: 마진이 충분히 크면서(質 벽 회피) baseline 107s보다 빠른(cap5 탈락) **중간 속도**.
- 비유: cap10=한계속도 풀스로틀(스핀), cap5=초보 저속(안전하나 굼뜸). **중급 운전자(적당 속도+여유)** 가 필요.

---

## 2. 목표 재명확화 (성공기준 분리, 016 D6)
- **(가) 1차 산출물 = baseline 107s 초과 안전완주** = "모방(BC)". 본 처방의 목표.
- **(나) stretch = <56s** = "stitching"(011이 "확률 낮음"). value 단계(Phase 4) 몫.
- cap5는 완주해도 115s라 (가)에 미달 → **cap5 속도대는 답이 아님**. cap10 속도대(56s)를 마진 있게 내야 함.

---

## 3. 처방 핵심 — v_max 낮춘 rollout = 새 정책 없이 마진 데이터
- 보유 정책: cap5/cap10/cap15/cap20 ckpt만(`runs/cap10_oschersleben/step_45k.pt` 등). **cap7-8 중간
  정책은 없음.**
- ★ `collect_crash_data.py`는 `--ckpt <정책> --v_max <캡>` 구조(주석 L26-33). v_max는 정규화 action을
  raw m/s로 푸는 스칼라(L138). **cap10 정책을 `--v_max 8`로 rollout하면 같은 정책이 0.8배 속도로 도니
  마진↑** — 새 정책 학습 불필요.
- 예상: cap10 lap ~28.6s → cap8 ~35.7s(×10/8) → **2랩 ~71s, baseline 107s 초과** ✓.
- ★ 수집은 **CPU**(`config.device='cpu'`, 주석 L21) → GPU 무관, 다른 학습과 병행 가능.

---

## 4. 실행 순서 (Phase 게이트)
| Phase | 무엇 | 가르는 것 / 산출 | 비용 |
|---|---|---|---|
| **1. cap8 타진** ★착수 | cap10 정책 `--v_max 8`로 **20ep rollout** → 완주율·lap time 확인 | "v_max 낮추면 마진 생겨 완주? 속도 baseline 초과?" | ~20-30분(CPU) |
| **2. cap8 본수집** | (타진 OK면) cap8 완주 데이터 50-100ep 수집 | 마진 데이터 확보 | ~1-2h(CPU) |
| **3. cap8 prior(256) → BC** | lidar256 obs로 prior 학습(촘촘 ckpt) → BC 평가(K1/K3) | **마진이 완주를 여나?** | ~1.5h+ |
| **4. value stretch** | (완주 후) cap8-stats γ0.999 value → scale 스윕 | <56s 시도(낮은 확률) | 고비용 |

**판정 분기(Phase 3 후)**:
- ✅ **완주 나오면** → 마진이 답. cap8이 sweet spot → Phase 4.
- ❌ **안 나오면** → covariate도 필요(cap5처럼) → §5 A(시작점 다양화) 추가.

---

## 5. 고려 중인 방안들 (대안·보강)
| 방안 | 언제 | 비고 |
|---|---|---|
| **A. 시작점 다양화(복구 covariate)** | Phase 3 실패 시 | cap8도 covariate 벽이면. 단 어댑터 `reset(options)` 글루 선행(016 D5) — 코어 인접, 사용자 승인 |
| **B. v_max 스윕(7/8/9)** | Phase 1 결과 보고 | 8이 애매하면 7(더 마진)·9(더 빠름)로 sweet spot 탐색 |
| **C. 새 중간 캡 정책 학습** | v_max rollout이 부자연스러우면 | RL_project Dreamer 학습 ~수시간(최후) |
| **D. obs 512** | 보조 짜내기 | 質 근본이라 효과 제한적(보류, 015 §5) |
| **E. value(γ0.999)로 質 교정?** | — | **불가**: value는 속도만 보상(011 §2), 감속(마진) 못 시킴 → Phase 4는 stretch용이지 質 해결 아님 |

---

## 6. 정직한 리스크
- **복합 원인이라 마진만으론 부족할 수 있다**: cap5가 마진 충분한데도 covariate 벽에 막혔듯, cap8도
  그럴 수 있다. → **Phase 1 타진(20분)이 가장 싸게 이를 가린다.**
- **v_max 낮춘 rollout이 "코너 감속"이 아니라 "전체 저속"** 일 수 있다(cap10 정책은 풀스로틀 학습). 그럼
  cap5와 성격이 비슷(저속 무감속)해져 covariate 벽 재현 가능 → A 필요.
- **BC 모방이면 56s 초과여도 stitching 아님**(016 D6). baseline 초과 완주를 (나)로 포장 금지.

---

## 7. critic 검수 포인트 (Phase 1~3 후 재검수용)
- **F1**: v_max=8 rollout이 정책 학습 캡(10)과 달라 정책이 **비정상 거동**(예: 항상 +1 풀스로틀이라 8 일정)
  하지 않나? 그럼 "마진"이 아니라 단지 cap5형 저속 재현일 뿐.
- **F2**: cap8이 완주해도 그게 "마진 덕"인지 "단지 느려서(cap5처럼)"인지 분리 가능한가? (lap time이
  baseline 초과인지가 관건)
- **F3**: Phase 3 BC가 완주해도 cap8 데이터도 단일 라인(시작 pose 고정)이면 covariate 벽이 남아 부분
  완주만 될 위험(NR3 재발).
- **F4**: 256 obs를 cap8에 쓰는 게 정합한가(cap8 데이터로 normalizer 재fit, F1TENTH_MODE 신규 모드 필요).

---

## 8. 함정·규약·참조
- 수집=CPU(GPU 무관), 학습·평가=GPU run_in_background, kill 단독, 정지 전 state_N>0 확인([[verify-before-kill]]).
- 평가=RL_project .venv·cwd=vendor/diffuser·V_MAX20·2랩, F1TENTH_MODE=<학습모드> normalizer 정합, lidar256은
  학습·평가 둘 다 `F1TENTH_LIDAR_DOWNSAMPLE=256`.
- 출력 절대경로, `run_logs/`·`runs/crash_data/` 보존, 촘촘 ckpt(save_freq2000/n_saves20).
- cap8 prior 평가 시 scale=0이면 더미 guide로 value 우회(017 0b fix, plan_f1tenth.py build_policy).
- 참조: 진단 [[017-diagnose-first-rebaseline-after-016]] / 검수 [[016-adversarial-critic-of-015-diversification]] / SSOT [[011-staged-gate-plan-v2-finalized]]
- 코드: `f1tenth_RL_project/scripts/collect_crash_data.py`(--ckpt/--v_max L26-33, device cpu L21, save L186-231) ·
  정책 `runs/cap10_oschersleben/step_45k.pt` · 로더 `vendor/diffuser/diffuser/datasets/f1tenth.py`
- 로그: 신규 `runs/crash_data/cap8/` · `run_logs/cap8_*`

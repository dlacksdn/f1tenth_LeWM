# 019 — Diffuser 모델 부적합 결론 → 정책 기반 Offline RL로 전환 (세션 인수인계)

> 2026-06-21. 이 세션의 종착점이자 다음 세션 시작점. cap8+jitter 처방까지 시도해 **모든 데이터 처방이
> BC 완주 0**임을 확인하고, 병목이 **데이터가 아니라 Diffuser 모델 자체의 한계**임을 진단, 과제 PDF로
> **모델 선택이 자유**임을 확인해 **정책 기반 Offline RL로 전환**을 결정. append-only. 엄밀(수치·파일) +
> 쉽게(표). git/구현 미실행(기록만). 컨텍스트 풀(full) → 새 세션 인계용.

---

## 0. 한 줄 (가장 중요)
**Diffuser(diffusion planner)는 f1tenth 고속 실시간 제어에 구조적으로 부적합**하다 — 같은 cap8 완주
데이터로 **RL 정책은 9.5% 완주, Diffuser BC는 0% 완주**. 마진·covariate·obs를 다 조정해도 BC 완주
0이었던 건 데이터가 아니라 **생성 planner의 모델 한계**(느린 추론·open-loop·생성 노이즈) 때문.
과제 PDF(p6-7)는 **"Offline RL"만 요구(알고리즘 자유)**, 목표는 "기존 policy 대비 빠른 lap time".
→ **결정: Diffuser 폐기, 정책 기반 Offline RL(IQL / TD3+BC / CQL 등)로 전환.** 데이터·dreamer
인프라는 재사용, Diffuser 분석은 보고서의 "왜 planner는 부적합한가" 비교자료.

---

## 1. 이 세션이 한 일 (016 검수 이후)
016(015 검수)에서 "진단 먼저"로 재배열 → Phase 0 진단 → cap8 마진 처방 → cap8+jitter 결합까지:
1. **Phase 0 진단**(017): cap10 BC 폐루프 덤프(0a) + obs 256 프로브(0b) → 병목을 質(margin0)+obs(보조)로
   좁힘, covariate 약화.
2. **cap8 처방**(018): cap10 정책을 `v_max=8`로 rollout = 새 정책 없이 마진 데이터(lat-demand 1.73,
   2랩 ~63s, baseline 초과). 타진 완주율 10%.
3. **시작점 다양화 글루**: 어댑터(`vendor/dreamerv3-torch/envs/f1tenth.py`)+wrappers(TimeLimit/UUID)에
   pose 주입 통로(하위호환, **dreamer 독립 검증 완료** — 무인자 호출 시 100% 동일, train env 스모크 통과).
4. **cap8+jitter 본수집**(완주 26ep, 시작 lidar std 0.037 = covariate 포함) → prior(256) 학습 → **BC 평가**.

## 2. 핵심 발견 (전부 실측, run_logs 보존)
### 2.1 모든 BC 완주 0 — 마진↔생존만 선형, 완주는 불변
| prior (BC scale=0) | lat-demand(마진) | covariate | 완주 | 생존 median |
|---|---|---|---|---|
| cap5 | 0.78 (큼) | 없음 | **0/10** | 349 |
| cap8+jitter | 1.73 (중간) | **있음** | **0/10** | 118 |
| cap10 | 2.26 (작음) | 없음 | **0/10** | 66 |
→ 생존은 마진(=1/lat-demand)에 **완벽 비례**, 그러나 완주는 **어느 것도 0**. 마진·covariate·obs(256)
어떤 데이터 처방도 BC 완주를 못 엶.

### 2.2 병목 = 첫 코너 質 벽 = Diffuser 정밀도 한계 (0a·cap8 덤프)
- 모든 BC가 **첫 코너까지 데이터처럼 추종(cmd_v 일치) → 첫 코너서 front 급감(0.04→0.01)+조향 포화 → 충돌**.
- 데이터(cap8 완주)는 코너를 front 0.024로 통과하는데 **BC는 0.01로 충돌** = **Diffuser가 좁은 코너
  실행을 데이터만큼 정밀 재현 못 함**(생성 노이즈).
- jitter(covariate)는 이걸 못 고침 — "코너 후 표류"가 아니라 **코너에서 먼저 죽기** 때문.

### 2.3 결정적 대비 (모델 한계의 직접 증거)
**cap8 정책(dreamer RL) 완주 9.5% ↔ 그 데이터로 학습한 Diffuser BC 완주 0%.** 데이터 동일, 모델만
바꾸니 완주 능력 소실 → 병목은 모델.

## 3. Diffuser ↔ f1tenth 고속제어 구조적 mismatch
| Diffuser 특성 | f1tenth 요구 | 결과 |
|---|---|---|
| 느린 추론(128-step를 20-step denoising) | 50Hz 실시간 | 본질적으로 느림(원논문 maze2d·locomotion=준정적) |
| open-loop 계획(전체 trajectory 1회 생성) | 매 순간 피드백 | compounding(우리가 본 covariate) |
| 생성 노이즈 | 마진0 코너 정밀 실행 | 첫 코너 충돌 |

## 4. 과제 PDF 확인 (p6-7, 모델 자유 확정)
- 주제 = **Offline RL** ("잘 달린 데이터로 학습해 더 잘 달리기"). **Diffuser 지정 없음.**
- 핵심 = "100초대 policy로 데이터 수집 → offline RL → **추가 상호작용 없이** 정책 개선".
- 기대 = "**기존 policy 대비 빠른 lap time**"(정량 성과).
- → Diffuser는 다른 세션 추천이었을 뿐. **알고리즘 자유**, 목표는 빠른 lap time.

## 5. 결정 — 정책 기반 Offline RL로 전환
| | Diffuser(폐기) | 정책 기반 Offline RL(전환) |
|---|---|---|
| 방식 | trajectory 생성(planning) | 정책(상태→행동) 직접 학습 |
| 추론 | 느림 | 빠름(1 forward)=실시간 |
| closed-loop | open-loop→compounding | 자연 피드백 |
| 노이즈 | 생성노이즈→정밀도↓ | 결정론적→정밀 |
- 후보 알고리즘: **IQL(Implicit Q-Learning)**, **TD3+BC**, **CQL** — 전부 정책 직접 학습(실시간 적합).
  Decision Transformer는 sequence model이라 Diffuser와 유사 한계 가능 → 후순위.
- 가장 단순 출발: **TD3+BC 또는 IQL**로 주어진 데이터(cap5/10/15/20) 학습 → 정책 → 평가.

## 6. 자산 (재사용 — 폐기 금지)
- **데이터**(`/home/dlacksdn/f1tenth_RL_project/runs/crash_data/`): cap5_full(완주22), cap10_full(완주30),
  cap15(371충돌), cap20(291충돌), cap8_jitter(완주26·시작다양), cap8_probe(타진). 동결.
- **정책 ckpt**(`runs/{cap5,cap10,cap15,stage2}_oschersleben/`): cap5/10/15/20. cap10=step_45k(lap26.2s).
- **dreamer 인프라**(`vendor/dreamerv3-torch`, `dreamer_f1tenth`): RL 학습/env. offline RL 구현 시 env·데이터
  로더 재사용 가능.
- **Diffuser**(`vendor/diffuser`): 폐기하지 말고 보존 — 보고서 "planner 부적합 분석" 자료 + 학습된 prior들
  (`logs/f1tenth/diffusion/f1tenth_{cap10,cap5,cap8_256,...}`).
- **평가 인프라**: `vendor/diffuser/scripts/plan_f1tenth.py`(diffuser용); offline RL은 별도 평가 필요(eval_gate
  재사용 가능, `f1tenth_RL_project/scripts/eval_gate.py`).
- **run_logs/**: 모든 학습·평가·진단 로그(보고서 그래프 재료, 삭제 금지).
- **_thinking**: plan_new/009~019(계획·검수), implementation/013~015(P6·P7 진단).

## 7. 다음 세션 시작점 (구체 작업은 사용자 지시 대기)
1. **offline RL 알고리즘 선택·구현/통합**(IQL/TD3+BC) — 주어진 데이터로 정책 학습.
2. **순수 offline 제약 준수**: 과제 "추가 상호작용 없이" → 주어진 cap 데이터로만 학습(cap8/jitter
   추가수집은 진단용이었고 본 과제엔 긴장 — 새로 수집하지 말고 기존 데이터 우선).
3. **평가**: 학습 정책 → f110 Oschersleben 2랩 → lap time vs baseline(107s). 빠르면 과제 목표 달성.
4. **보고서**: Diffuser 비교(왜 planner는 0%, 정책은 완주) = 강한 분석 축.

## 8. 절대 규칙 (변경 불가 — 새 세션 필수 준수)
- ★ `git add/commit/push/pull`은 **사용자 지시 시에만**(자율 금지). 코드 구현·`_thinking` 저장도 명시 지시 시.
- ★ **로그·모델 폐기 금지**, 모아둘 것(CLAUDE.md). run_logs·diffuser ckpt·데이터 보존.
- ★ **dreamer ↔ diffuser/offline 독립**: dreamer 공용 코드 수정 시 하위호환+스모크 검증(메모리 [[dreamer-diffuser-independence]]).
- ★ 코어 무변경(temporal/diffusion/trainer); GPU=run_in_background, kill 단독(복합+kill=exit144), 정지 전
  state_N>0 확인, 평가=RL_project `.venv`, V_MAX20·2랩·Oschersleben.
- ★ `_thinking` append-only, 명시 지시 시만 저장/읽기. **한글로 답변.**

## 9. 참조
- 직전 검수: [[016-adversarial-critic-of-015-diversification]] / 처방 계획: [[017-diagnose-first-rebaseline-after-016]] · [[018-margin-data-via-vmax-rollout-plan]]
- 데이터 진단 흐름: [[012]]→[[013]]→[[014]]→[[015]] / SSOT(구): [[011-staged-gate-plan-v2-finalized]]
- 로그: `run_logs/{a3_cap10_K3, cap5_bc_K*, cap8_bc_K*, cap8_bc_dump, cap10_256_bc_K*}.json` · `cap8_jitter_collect.log` · `train_cap8_256.log`
- 과제: `_thinking/raws/AIE4003_RL_F1TENTH.pdf`(p6-7 = Offline RL 주제·목표)
- 코드: 로더 `vendor/diffuser/diffuser/datasets/f1tenth.py`(모드 all/complete/cap10/cap5/cap8) · 수집 `f1tenth_RL_project/scripts/collect_crash_data.py`(--start-jitter 추가) · 어댑터/wrappers(pose 통로)

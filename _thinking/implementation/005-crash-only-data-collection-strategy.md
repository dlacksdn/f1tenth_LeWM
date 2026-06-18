# 005 — cap-10 baseline 확정 + 충돌(crash)-only 데이터 수집 전략 확정

> 2026-06-19. [[004-cap5-baseline-and-data-design-pivot]] 후속. cap-10 baseline 확정 +
> P2 데이터 수집 전략을 **충돌-only**로 확정(사용자 결정) + deterministic 문제 1차 소스 해결 +
> 구현 사양. append-only. 선행: [[004-cap5-baseline-and-data-design-pivot]], [[002-p1-speedcap-policies-and-rlproject-changes]].

---

## 0. 한 줄 상태
**P1 완료**(cap-5/10/15 봉우리 deterministic 2랩 완주 확정). cap-10 baseline=`step_45k`(2랩 53.66s, 무진동).
**P2 전략 확정 = cap-5/10/15/20 "충돌 데이터만" 수집**(완주 미수집), stochastic rollout, pose+v_max 추가.
다음 = 수집 스크립트 구현 → smoke → 4 tier 동시 대량 수집.

## 1. cap-10 baseline 확정 = `step_45k.pt`
- deterministic eval(`eval_gate --v_max 10 --episodes 5`): step_45k/40k/35k/30k **전부 완주율 1.0**(cap-5와 대조, 무진동).
- **채택 `runs/cap10_oschersleben/step_45k.pt`: 2랩 53.66s(27.3+26.36), 완주율 1.0, A12/A13 PASS.**
- ★ **cap-10(V_MAX=10)=학습 sweet spot**: cap-5(저속)·cap-15(거의 무캡 고속)는 진동, cap-10은 무진동(전 스냅샷 완주). 보고서 관찰거리.
- watch에서 본 ~60s는 stochastic, deterministic(argmax)은 53.66s(cap-5와 같은 현상).

## 2. tier 스펙트럼 완성 (P1)
| tier | 정책 | 2랩 | 역할 |
|---|---|---|---|
| cap-5 | step_25k | 107.16s | baseline(데이터 출처) |
| cap-10 | step_45k | 53.66s | 중간 |
| cap-15 | step_105k | 37.3s | 빠름 |
| cap-20 | stage2 best | ~36s | 최고속(무제한) |

## 3. ★ deterministic 문제 해결 (1차 소스 dreamer.py:96-113)
- **정체**: eval_gate가 `training=False`(→`actor.mode()`, argmax) + `eval_state_mean=True`(→latent도 `.mode()`) → 5/5 완전 동일.
- **해결**: 수집을 **`training=True`(→`actor.sample()`) + `eval_state_mean=False`(→latent sample)** 로 = 정책의 학습된 분포에서 자연 샘플. **dreamer가 train_eps 265개 다양성을 만든 그 메커니즘.** 같은 출발점도 매 ep 다른 궤적.
- ★ **action noise 인위 추가 불필요**(actor 자체가 stochastic dist, sample()이 자연 다양성 — 왜곡 없음). 사용자 noise 우려 원천 해소.
- 증거: cap-5 `eval_eps`(stochastic, pose 고정) = 완주 30 + 크래시 131, 라인/도달거리 다양. **pose 고정해도 stochastic 하나로 다양성 충분.**

## 4. ★★ 충돌-only 데이터 수집 전략 (사용자 확정)
- **충돌(collision) ep만 사용. 완주 데이터는 지금 안 모음**(나중에 P4 합성 실패 시 앵커로; 봉우리 정책 있어 언제든 분 단위 확보).
- **왜 충돌만이 옳은가**: 완주 데이터를 넣으면 Diffuser가 그걸 **모방(BC)**해도 완주 → stitching 증명 안 됨. **완주 데이터를 빼면 모방 대상이 없어 Diffuser가 충돌 조각을 value로 이어붙여 완주를 진짜 "합성"해야 함** = offline RL stitching의 순수 증명(사용자 직관).
- **★ 4 tier = cap-5/10/15/20** (cap-20 포함 — 이전 "cap-20 폐기"는 완주정책=expert 한정 실수):
  - 충돌 데이터에 **속도 5~20 스펙트럼 내장**. cap-20 충돌엔 고속(명령 20/실현 ~15) 구간 → **빠른 완주 합성의 재료**(Diffuser=in-dist, 데이터에 빠른 속도 없으면 못 냄).
  - **cap-20 충돌 ≠ expert 시연**(완주 안 함 = 실패 데이터) → 교수님 no-expert 위배 아님. 보고서에 "완주 시연 없이 충돌만" 정직 명시.
  - cap-20 충돌 = stage2 기존 259개 + 추가 rollout.
- **정책 선택**: 완주 봉우리(25k/45k/105k/stage2-best) + stochastic rollout → **충돌 ep만 저장**(완주 ~19% 폐기, 부산물). 봉우리의 충돌이 "거의 완주할 뻔한 실패" = 트랙 깊이 가서 **안전 구간 풍부 = 최상급 stitching 재료**(완주 못 하는 충돌 스냅샷보다 우수). 봉우리는 이미 학습 완료라 추가 비용 0.
- **behavior policy = cap-5(107s)**. 서사: *"cap-5/10/15/20이 충돌한(완주 못 한) 실패 데이터만으로, cap-5보다 빠른 2랩 완주를 Diffuser가 합성"*.

## 5. 데이터 항목 (Diffuser 계약, 1차 소스 [[006-glue-correction-and-data-contract]])
- **필수**: observations=concat(lidar,state), actions, rewards(reward/log_reward_*), terminals(is_terminal). timeouts=충돌-only라 전부 terminal → 전부 False, `assert not timeouts.any()` 통과. → **dreamer npz 자동 저장**(reward·log_reward_progress·collision 포함, 키 확인됨).
- **추가**: `pose`(npz 유일 갭, env에서 추출) + `v_max(tier)`(P3 역정규화 필수 — +1이 cap5=5·cap20=20m/s).
- ★ **lap 신호 부재 우려 약화**: Diffuser value용 D4 reward(006)는 **lap 보너스 제외**(progress dense + 축소 collision만). 충돌 데이터의 progress 신호만으로 value가 "완주=progress 최대" 방향 유도. 004 §3.2의 "R_lap 0번" 리스크는 약함.

## 6. 구현 사양 (scripts/collect_crash_data.py — 다음 구현)
- 기반: `eval_gate.py`의 make_env/load_agent/run_episode 재사용.
- **stochastic**: `eval_policy = partial(agent, training=True)` + `config.eval_state_mean=False`.
- **pose 추출(env 무수정)**: 모든 wrapper가 `gym.Wrapper` → `env.unwrapped._raw_obs["poses_x"/"poses_y"/"poses_theta"][0]` (f1tenth_env.py:290/469에서 step마다 갱신).
- **transition 정렬**: `tools.simulate` 미러(tools.py:192-199) — `transition = (step후 obs).copy()` + `action`(그 obs로 온 a) + `reward` + `discount`; 첫 transition은 action=0 패딩(add_to_cache:265). 기존 train_eps와 동일 형식 유지.
- **충돌 필터**: `info['cause']=='collision'` ep만 저장(`lap_complete` 폐기; diverged/reverse도 제외 — 비정상).
- **npz**: 기존 키(lidar/state/action/reward/log_*/is_terminal/is_last) + `pose`(T,3) + `v_max`(스칼라/(T,)).
- **동시 수집**: 12코어/12GB free → **4 tier CPU 동시**(config.device="cpu", GPU 무관 — eval_gate도 CPU). wall-clock 4배 절약. 각 tier 다른 출력 디렉토리.
- pose 변경(출발점)은 선택적 강화 — stochastic만으로 다양성 충분하니 v1은 default_pose 고정.

## 7. 미검증 리스크 (P4에서만 확인)
순수 충돌-only의 **유일한 남은 리스크 = value가 "고속이지만 충돌로 이어지는" 구간을 걸러내는가**(progress 높지만 collision으로 끝나는 궤적의 return이 낮아 회피되는지). 고속 레이싱은 antmaze보다 어려움. → P4 생성 궤적 품질에서 확인, 실패 시 완주 앵커 극소량 투입(단계적 escalate).

## 8. 다음 단계
1. `scripts/collect_crash_data.py` 구현 → smoke(1 tier 소수 ep: pose·충돌필터·npz 키 검증).
2. 4 tier(cap-5/10/15/20) 동시 대량 수집(tier당 수십~수백 충돌 ep, crash만).
3. P3: f1tenth 로더(각 tier v_max로 raw 역정규화 일관화) + normalizer + P0 글루 디커플 병행([[006-glue-correction-and-data-contract]]).
4. 분기마다 _thinking + commit + push. push: github.com/dlacksdn/f1tenth_planning_with_diffusion.

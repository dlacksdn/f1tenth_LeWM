# 006 — 봉우리 충돌 수집 1차 결과 + 충돌 스냅샷 보강 전환

> 2026-06-19. [[005-crash-only-data-collection-strategy]] 후속. 4 tier 봉우리 stochastic 충돌 수집의
> 1차 결과(봉우리의 역설) + cap-5/10 빈약 → 충돌 스냅샷 보강 재수집(사용자 승인). append-only.

---

## 0. 한 줄 상태
4 tier 봉우리 충돌 수집(005 실행): **cap-15/20 풍부(280/187), cap-5/10 빈약(13/19)** — 봉우리가 완주를
너무 잘 해서. → **cap-5/10을 충돌 스냅샷으로 보강 재수집**(cap-5=step_35k 확정, cap-10=선정 중),
봉우리분 13/19개는 "깊은 충돌"이라 보존·합침. cap-15/20 봉우리 유지(충분).

## 1. 봉우리 수집 1차 결과 (시작 후 ~7.7h, 1차 소스, 아직 진행 중)
| tier | 봉우리 정책 | 충돌 npz | 충돌률 | 완주 ep 길이 | 진행 |
|---|---|---|---|---|---|
| cap-5 | step_25k | **13** | 18% | ~5751 step | 73/500 |
| cap-10 | step_45k | **19** | 14% | ~2833 step | 139/500 |
| cap-15 | step_105k | **280** | 75% | ~1932 step | 375/500 |
| cap-20 | stage2 best(FULL) | **187** | 60% | ~722 step | 311/500 |

## 2. ★ 봉우리의 역설 (진단)
- **잘 완주하는 tier(cap-5/10)일수록 충돌 데이터가 안 모인다**: stochastic에서도 완주율 82~86% → 충돌 빈약 +
  완주 ep가 길어(cap-5 5751 step!) 폐기 비용까지 커 진행이 거북. 현 속도면 cap-5는 500 ep까지 **~45h**(비현실).
- **진동/고속 tier(cap-15/20)는 충돌 잘 남 + ep 짧아** 빠르게 풍부해짐.
- 결론: **"충돌-only + 봉우리"는 안정 tier에서 데이터가 안 모인다.** 어제 워크플로우가 짚은 위험2가 현실화.
  사용자 직관("봉우리라 그럴 수도")이 데이터로 확인됨.

## 3. 전환 결정 (사용자 승인 = 옵션 1)
- **cap-5/10을 "충돌하는 이른 스냅샷"으로 재수집**(= 사용자 원래 "충돌 정책" 방향):
  - **cap-5 = `step_35k`**(deterministic 1랩도 못 감 = 100% 충돌, ~1183 step 짧음 → 빠른 대량 수집). 확정.
  - **cap-10 = step_15/20/25k eval로 선정**(완주율<0.5 + 즉사 아닌 것). ← Workflow `cap10-crashsnap-select` 진행 중.
- **봉우리 충돌 13/19개는 보존·합침**: "거의 완주할 뻔한 실패"라 긴 안전 구간 = 깊은 stitching 재료.
- **cap-15/20 봉우리 유지**(충분, 곧 500 완료).
- **최종 데이터셋 구성**: cap5(봉우리 13)+cap5_lowsnap / cap10(19)+cap10_lowsnap / cap15(280) / cap20(187).
  → 속도 5~20 스펙트럼 + 깊은(봉우리)·얕은(스냅샷) 충돌 혼합.
- 출력 디렉토리: `runs/crash_data/{cap5,cap10,cap15,cap20}`(봉우리), `{cap5_lowsnap,cap10_lowsnap}`(스냅샷).

## 4. 충돌 스냅샷 trade-off (인지)
충돌 스냅샷 = 얕은 충돌(일찍 죽어 트랙 후반 커버 약함), 봉우리 = 깊은 충돌(트랙 깊이 주행). **둘을 합쳐 보완**
(스냅샷이 양·시작부 커버, 봉우리가 깊이·후반부 커버).

## 5. ★ P3 로더 주의 (어제 workflow 검증서 발견 — 잊지 말 것)
- **max_path_length**: 봉우리 충돌 ep가 길 수 있음(TimeLimit 9000 env-step 상한, 봉우리는 거의 완주). P3
  로더 `ReplayBuffer(max_path_length=...)`를 **실측 max 이상**으로(과거값 1881로 잡으면 buffer.py:66 assert 터짐).
- **timeouts**: npz에 timeouts 키 없음 → 로더가 `timeouts = is_last & ~is_terminal` 합성(충돌 ep는 전부 False
  → buffer.py:79 assert 통과). config `termination_penalty=None` 병행.
- **pose**: `log_pose_theta`는 **unwrapped world yaw**(누적, [-π,π] 아님), 충돌 step은 theta=0(f110 sim
  artifact). pose는 Diffuser observations(=concat(lidar,state)=1085D)에 **미포함**(별도 채널)이라 normalizer
  무영향. 단 P3에서 pose를 centerline 피처로 쓰면 wrap-to-[-π,π] + 충돌 행(theta=0) 제외 필요.

## 6. 데이터 항목·정렬 (005 §5-6 확정 재확인)
npz = 기존 train_eps 17키(lidar/state/action/reward/is_*/log_*/discount/logprob) + pose(T,3)/v_max(T,)/
log_pose_*. transition 정렬 = tools.simulate 미러(action[0]=0 패딩, obs[t]=action[t] 결과). 충돌(collision)
ep만 저장. (코드: scripts/collect_crash_data.py + f1tenth_env.py log_pose, RL_project master c8f980c commit됨.)

## 7. 다음
cap-10 충돌 스냅샷 선정 완료 → cap-5(step_35k)/cap-10(선정) 재수집 시작 → tier별 충돌량 균형 확인 →
P3 f1tenth 로더(tier별 v_max 역정규화 + timeouts 합성 + max_path_length 실측) + P0 글루 병행. push: github.com/dlacksdn/f1tenth_planning_with_diffusion.

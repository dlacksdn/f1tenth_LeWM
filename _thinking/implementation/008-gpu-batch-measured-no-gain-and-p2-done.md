# 008 — GPU 배치 측정(이득 없음·가설 반증) + cap-10 재수집 완료 = P2 종료

> 2026-06-19. [[007-p2-collection-state-and-gpu-batch-handoff]] 후속. 007이 다음 1순위로
> 지목한 "GPU envs=N 배치 collect"를 **구현·품질검증·속도측정**한 결과 **throughput 이득
> 없음(가설 반증)** → 단일 CPU로 cap-10 완주+충돌 재수집 완료 → **P2 데이터 수집 4 tier
> 전부 종료.** append-only. 선행: [[007-p2-collection-state-and-gpu-batch-handoff]],
> [[006-peak-crash-yield-and-snapshot-pivot]], [[005-crash-only-data-collection-strategy]].

---

## 0. 한 줄 상태
**P2 완료**(cap-5/10/15/20 데이터 확보). GPU 배치는 구현·품질검증까지 했으나 **측정상
throughput 이득이 없어**(병목=f110 env.step, behavior 모델이 작아 추론 비병목) **미채택**(코드
보존). cap-10은 단일 CPU로 재수집(완주30+충돌10). 다음 = **P3 로더 + P0 글루**.

## 1. ★ GPU 배치 측정 결과 (1차 소스, 가설 반증)
007 §3은 "추론은 ep마다 독립 → envs=N 배치로 N배 처리량"을 가정. **직접 측정이 이를 반증:**
cap-15 step_105k, episodes=12, overhead(episodes=0 wall) 분리, Σ(처리 step)/rollout-s 기준.

| 구성 | rollout | Σstep | **step/s** | vs CPU단일 |
|---|---|---|---|---|
| **CPU envs=1** | 54.6s | 12251 | **224.5** | 기준 |
| GPU envs=8 Damy | 27.1s | 5031 | **185.5** | **−17%** |
| GPU envs=8 Parallel | 28.1s | 6706 | **238.7** | **+6%** |

- **Damy(GPU 추론 배치, env.step 메인 직렬)는 단일 CPU보다 느림**(−17%): GPU↔CPU 전송 +
  커널 런치 오버헤드 > 작은 모델 배치 추론 이득.
- **Parallel(env.step 8프로세스 병렬)조차 +6%**: IPC(1080-D lidar pipe 전송) 오버헤드가 상쇄.
- ★ 근본 원인: ① behavior 모델이 작아(Dreamer actor; **Diffuser 평가의 1D U-Net과 전혀 다름**)
  추론이 병목이 아님 ② 진짜 병목 = f110 `env.step`(Python sim) ③ CPU torch BLAS가 이미
  멀티코어(~5코어) 활용 → "단일 env라 GPU가 비효율"이라는 007 전제가 **이 규모에선 불성립**.
- **결론: GPU 배치 미채택.** 단 코드는 보존(`--envs`/`--device`/`--parallel` + `collect_batch`).
  더 큰 모델/다른 수집엔 유효. (cf. Diffuser **학습/평가**는 GPU 필수 — 이건 *데이터 수집* 한정 결론.)

## 2. collect_crash_data.py 변경 (commit 대상)
- **배치 모드 추가**: `--envs N`(>1=`collect_batch`, `tools.simulate` L150-205 배치 루프 미러),
  `--device cpu|cuda`, `--parallel`(Damy↔Parallel). **`--envs 1`=기존 단일 경로 100% 불변**(regression 0).
- **품질 동일성 실측**(단일 CPU vs GPU 배치 npz): 키셋 완전 동일, dtype+비시간축 shape 일치,
  0-패딩 정렬(action[0]=logprob[0]=0), pose(T,3), v_max(tier), 충돌필터, 종료플래그 동일.
  (stochastic이라 정확 궤적은 다르나 **구조/계약 동일** — 사용자 검증 기준 충족.)
- **부수 버그 수정**: `--save-complete` help의 `%` 미이스케이프 → `--help` 시 argparse 크래시
  (`100% → 100%%`). 실수집(정상 인자)엔 무영향이라 034 검증 때 미발견.

## 3. ★ P2 데이터 최종 현황 (1차 소스, `runs/crash_data/`)
| tier | 완주(2랩) | 충돌 | 디렉토리 | 정책(--v_max) |
|---|---|---|---|---|
| **cap-5** | **22** | **22** | cap5_full(완주22+충돌9) + cap5(충돌13) | step_25k (5) |
| **cap-10** | **30** | **31** | cap10_full(완주30+충돌10) + cap10(충돌21) | step_45k (10) |
| **cap-15** | 0 | **371** | cap15 | step_105k (15) |
| **cap-20** | 0 | **291** | cap20 | stage2 best (20) |
- **전략 정합(005/007)**: 저속~중속(cap-5/10)=완주+충돌(저속 완주=진정성=behavior 시연),
  고속(cap-15/20)=충돌만(BC 회피 + reward 쏠림 방지). 총 완주 52 / 충돌 715.
- cap-10 재수집 명령: `--ckpt cap10_oschersleben/step_45k.pt --v_max 10 --save-complete
  --max-env-steps 9000 --out runs/crash_data/cap10_full --episodes 40` → 완주30+충돌10(수집률 1.0).
  (stochastic 완주율 75% 실현 = 006 예측 86% 부근, 표본 변동.)

## 4. ★ P3 로더 근거 1차 검증 (완주/충돌 종료플래그)
cap10_full 완주 ep로 **006 §5 "timeouts 합성"을 실측 확정:**
- **완주 ep**: `is_terminal[-1]=False`, `is_last[-1]=True`, `log_completed[-1]=1`, lap_time_s>0 2개(2랩).
  → `timeouts = is_last & ~is_terminal` = **1(마지막 step만 True)**. 완주는 **non-terminal**(MDP
  종료 아님 = Diffuser value 부트스트랩 가능) + truncation(timeout)으로 표기됨.
- **충돌 ep**: `is_terminal[-1]=True` → `timeouts = 0`. 충돌은 terminal.
- → P3 로더는 `termination_penalty=None` + `timeouts=is_last&~is_terminal` 합성이면 충돌·완주
  혼합을 정확히 처리(`buffer.py:79 assert` 통과). 완주 ep len=2807(긴 ep) → `max_path_length`는
  실측 max 이상(과거 1881로 잡으면 assert 터짐, 006 §5).

## 5. 다음 작업 (순서)
1. **commit**(collect 코드 + 008 + RL_project 035) — push는 사용자 지시 시만.
2. **P3 f1tenth 로더**: ① tier별 `v_max`로 raw 역정규화(+1이 cap5→5/cap20→20m/s 일관화)
   ② `timeouts = is_last & ~is_terminal` 합성(termination_penalty=None) ③ `max_path_length`=
   실측 max 이상(완주 ep ~2807) ④ pose `log_pose_theta`=unwrapped yaw(centerline 피처 시
   wrap-to-[-π,π] + 충돌 step theta=0 제외) + normalizer.
3. **P0 글루 디커플**(새 .venv, Diffuser repo) + 1-step([[006-glue-correction-and-data-contract]]).
4. (선택) cap-5 완주 보강 — cap-10이 완주30이라 cap-5(완주22)와 균형 OK, 현재 충분 판단.

## 6. 제약/규약 (불변)
★ push는 사용자 지시 시만(자율 push 금지). commit+문서는 분기마다 자율. 자율 pull 금지.
env 무변경(log_pose 진단키만). reward 독립(P4 재구성). 모든 lap=2랩. 트랙=Oschersleben.
venv: RL_project `.venv`(수집), Diffuser `.venv`(학습/평가, py3.8).

# 013 — P6 평가 실패·진단 + 다음 계획(주행 prior + 충돌 value) + 데이터 효율 분석

> 2026-06-20. [[012-p5-plan-impl-and-venv-integration]] 후속. P6 1차 평가가 **완주 실패**한
> 결과와 체계적 진단, 그로부터 도출한 **다음 계획(완주/주행 데이터로 prior 먼저, 충돌은 value로)**,
> 그리고 **"데이터를 더 모을 값어치가 있나"** 효율 분석을 1차 소스로 정리한다. **다음 세션 인수인계용.**
> append-only.

---

## 0. 한 줄
**현재 계획(전체 crash-위주 데이터로 diffusion+value 학습)은 닫힌 루프 완주 실패**(전 설정 충돌). 근본원인=
**crash-위주 데이터(715충돌/52완주)가 생성 prior를 오염**→플랜이 crash-prone. **다음=역할 분리: diffusion
prior는 "주행 궤적만"(완주 + 충돌데이터서 잘라낸 고속 lap-1) / value는 전체데이터.** **데이터: 추가 수집 전에
*기존 충돌 데이터에서 고속랩 249개를 공짜로 추출*하는 게 최우선**(재수집은 그 후 판단).

---

## 1. 현재 계획 실패 (P6 1차 평가, 자율 실행)
학습 완료(diffusion 80k loss 0.003 + value 100k **corr 0.98**) → plan_f1tenth 평가.
- **결과: 5/5 충돌, 완주 0, baseline 107.16s 미달.** (랩 자체를 못 끝냄.)
- **튜닝 전 설정 전부 충돌**: K ∈ {1,2,3,5,7,10,30}, scale ∈ {0,0.1,0.3,1.0}, best-of-32 value 선택 → 모두 실패.
  - K=1: 매 step 재계획 → 감속→과조향→**스핀/후진**(mean cmd −4.0 m/s). 실제 관람(WSLg)로 "제자리 주행" 확인.
  - K≥10: 모델의 빠른 플랜(16-17 m/s) 따라가나 **open-loop blind로 ~84 step(1.7s)에 코너 충돌**.

## 2. 체계적 진단 (무엇이 문제가 *아닌지* 배제 → 진짜 원인)
1차 소스(스크립트 `/tmp/p5_gen_diag.py`·`p5_cl_diag.py`·`p5_k_test.py`·`p5_value_test.py`)로 배제:
- **D3 아님**: scale 0/0.1/0.3/1.0 다 충돌(value 강화·best-of-N도 무효).
- **모델 아님**: 알려진 *training* obs를 넣으면 데이터와 거의 정확히 일치하는 action 생성 —
  step0 데이터 2.5↔생성 2.6, step300 9.19↔9.39, step1000 10.0↔9.50 (생성 std 0.02).
- **normalizer 아님**: action round-trip 완벽(mins/maxs [-0.419,-5]/[0.419,20]).
- **obs 구성 아님**: eval `env.reset()` obs ≡ training 시작 obs **byte 동일**(state 0, lidar128 차 0.0).
- **★ 진짜 원인 = closed-loop compounding error + crash-poisoned prior**:
  - 닫힌 루프 trace: t0-8 정상 가속(8 m/s)→t12-25 *이유 없이 감속*→t25-55 과조향→t55-75 후진/스핀
    (obs는 t65까지 in-dist; 즉 obs가 먼저 망가져서가 아니라 *정책이 스스로 degenerate*하고 그 뒤 분포 이탈).
  - 데이터의 93%가 충돌이라 모델의 "전형적 궤적 = 충돌로 끝남" → **생성 플랜이 crash-prone**(K=30이 84 step에
    충돌 = 플랜 자체가 주행을 못 버팀). value guidance는 이 prior를 못 이김.
- **→ 계획 critic D1/D2 우려가 실증됨**: crash-위주 데이터는 닫힌 루프 주행 학습에 너무 얇음.

## 3. 다음 계획 — 역할 분리: "주행 prior + 충돌 value"
사용자 직관("BC 먼저 + 충돌 입히기")의 원리적 버전. **실패 원인(prior 오염)을 정확히 피한다.**
- **diffusion(생성 prior) = "주행 궤적만"**: 완주(2-lap) + **충돌데이터에서 잘라낸 고속 lap-1**(§4). 충돌 궤적을
  prior에 넣지 않는다 → 생성 플랜이 *주행적*(non-crash) → 닫힌 루프가 주행 manifold에 머묾.
- **value = 전체 데이터(충돌 포함)**: "무엇이 빠르고 무엇이 충돌로 가는지"를 학습 → return 신호.
- **guidance** = 주행 prior를 *더 빠른* 쪽으로(value가 충돌 영역 밖에서 속도 유도). scale 튜닝 필요(너무 세면 충돌속도로).
- 기대: 깨끗한 주행 prior + 충돌-인지 value → **완주(>56s 안전) + 가능하면 그 이상(고속랩 재료로 stretch)**.
- ★ 핵심: **충돌 데이터는 value로만, prior엔 절대 안 넣는다**(지금 실패의 직접 원인이 prior 오염이므로).

## 4. ★ 데이터 효율 분석 — "더 모을 값어치가 있나"
**결론: 추가 수집 전에, 이미 가진 데이터를 다 짜내라. 그게 가장 효율적이고 거의 공짜다.**

### 4.1 ★ 공짜 고속 재료 — 충돌 데이터에서 lap-1 추출 (1차 검증 완료)
원래 "고속 tier=충돌-only" 전략이 cap-15/20 에피소드를 *통째로 충돌로 버렸다.* 하지만 그 안의 **lap-1 부분은
깨끗한 고속 랩**이다(2랩 돌다 충돌한 것뿐). `log_lap_time_s>0` 첫 시점까지 자르면:

| tier | lap-1 완주 ep | lap time | 잘린 길이 | 충돌 전무 |
|---|---|---|---|---|
| cap15 | **132**(36%) | 19.7s | ~988 step | **132/132** ✅ |
| cap20 | **117**(40%) | 18.0s | ~903 step | **117/117** ✅ |
| cap10_full | 37(92%) | 28.6s | ~1432 | 37/37 ✅ |
| cap5_full | 25(81%) | 57.9s | ~2896 | 25/25 ✅ |

→ **재수집 0으로 고속(18-20s) 깨끗 랩 249개** + 완주 52개 = **~300 주행 궤적**(속도 스펙트럼 18~114s 전부)을
prior용으로 확보. 지금 prior가 52완주로 얇아 covariate shift 났는데, **300으로 6배 + 고속 포함** → 큰 개선 기대.

### 4.2 수집 cost/benefit (값어치 순)
| 데이터 | 현재 | 마진 가치 | 비용 | 판단 |
|---|---|---|---|---|
| **lap-1 고속랩 추출** | 0(미추출) | **매우 높음**(고속+커버리지, 공짜) | **0**(기존데이터 처리) | **★최우선** |
| 충돌(value용) | 715 | 거의 0(이미 포화) | — | **수집 금지** |
| cap-10 완주(prior 커버리지) | 30 | 중간(추출풀로 충분할 수도) | 싸다(완주율 75%, ~1h/170개) | 추출 재학습 후 부족하면 |
| cap-5 완주(저속 커버리지) | 22 | 낮음(느림) | 싸다 | 후순위 |

- **데이터는 늘 많을수록 좋지만**, 지금은 **추출로 6배 늘릴 수 있어 *수집의 마진 가치가 일시적으로 낮다.***
  추출+재학습 후에도 닫힌 루프가 불안정하면, **그때 cap-10 완주 + cap-15/20 추가(고속랩 더)를 타깃 수집**(완주율
  높아 효율적, ~1-2h). 충돌은 더 안 모은다(포화).
- **효율 원칙**: ①기존 추출(공짜) → ②재학습·평가 → ③부족분만 타깃 수집. "전부 더 모으기"는 비효율.

## 5. 달성 가능한 두 층위 (정직)
| 층위 | 방법 | 결과 | 의미 |
|---|---|---|---|
| **안전** | 주행 prior(완주 중심) | ~56s(cap-10급) | baseline 107s 초과(문자적 성공), **모방**(개선 아님) |
| **야심** | 주행 prior(+고속랩) + 충돌-value | <56s 노림 | **진짜 개선**(불확실, stitching 베팅) |
- BC만으로는 behavior policy(cap-10)보다 빨라지지 않음(사용자 지적 정확). 진짜 개선은 §3 역할분리 + §4.1 고속랩에 달림.
- 어느 경로든 **"전체데이터 prior가 왜 실패했나"의 진단(§2)은 보고서 산출물로 이미 확보.**

## 6. 구현 메모 (다음 세션)
- **로더 변경**: ①prior용 "주행-only" 데이터셋 = 완주 ep + 충돌 ep의 lap-1 truncate(`log_lap_time_s>0` 첫
  시점까지, terminal 제거→timeout 처리). ②value용 = 전체 데이터 그대로. 두 로더(또는 env var 플래그)로 분리.
  (현 `f1tenth.py`는 단일 로더 — `F1TENTH_MODE=driving|all` 같은 플래그 추가가 깔끔.)
- diffusion 재학습(주행 prior, ~5-10h) + value 재사용 or 재학습(전체, 현 corr 0.98 ckpt 그대로 써도 됨).
- ★ **normalizer 정합(중요 함정)**: value guidance는 *diffusion이 생성한 궤적*(diffusion normalizer 공간)을
  value 모델이 평가한다. value 모델은 x를 내부 재정규화 없이 자기 학습 normalizer 공간으로 가정 →
  **prior와 value의 normalizer가 다르면(주행-only stats vs 전체 stats) value 그래디언트가 망가짐.**
  `check_compatibility`는 타입만 검사(stats 미검사)라 조용히 통과 → 안 잡힘. **해결: 전체데이터 normalizer로
  통일**(주행 prior도 그 stats로 정규화) 또는 prior/value가 같은 normalizer 공유. **value 재사용하려면 필수.**
- 평가는 plan_f1tenth.py 그대로(검증됨). K·scale 재튜닝.
- ★ 평가 venv=RL_project .venv(tap·GitPython 설치됨), cwd=vendor/diffuser, V_MAX=20. (실패 아님, 인프라 정상.)

## 7. 제약/규약
★ git add/commit/push 사용자 지시 시만. 코어 무변경(ValueFunction 예외). normalizer v1=frozen 데이터 재fit.
모든 lap=2랩. 트랙=Oschersleben. 데이터=RL_project/runs/crash_data(gitignore). 진단스크립트=/tmp/p5_*.py.
관람=`scripts/watch_f1tenth.py`(WSLg, --K로 모드).

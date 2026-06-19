# 011 — P4 학습 발사 + 적대적 이중검수(계획·구현) + 성공기준 재정의 + 데이터 다양성 검증

> 2026-06-19~20. [[010-p3-precision-vendor-move-valuefunction-fix]] 후속. P4 풀학습 발사 후,
> **계획 critic + 구현 critic을 각각 새 세션에서 적대적으로 돌려** 1차소스 재검증한 결과와
> 그에 따른 성공기준 재정의·수정을 기록. append-only. 상세 SSOT 정정은 [[008-diffuser-plan-v4]]
> 하단 append + [[007-p5-eval-infra-design]] 하단 정정 + 아키텍처 [[understand/001-diffuser-vs-value-architecture]].

---

## 0. 한 줄 상태
P4 diffusion+value 풀학습 발사(200k+200k 순차, ~38h, 무인). 계획·구현을 적대적으로 이중검수 →
**둘 다 (B)"수정 후 OK"**: 구현 code 버그 0, 계획은 성공기준만 정직하게 재정의. 데이터는 다양성·커버리지
실측으로 건전 확인. **유일 수정 = D5 config(discount) + P5설계 V_MAX=20**(둘 다 적용/기록). 다음 = P5 구현.

## 1. P4 풀학습 발사 (실측 신규 사실)
- **1차 게이트(10k) PASS**: loss 0.0048 수렴·폭주없음(SafeLimits+normed 효과). 내 kill로 step9100 정리.
- **★측정**: rate=**2.86 step/s**(GPU 95% **compute-bound** — dataloader 아님→num_workers 무효), 단일 학습
  GPU **7GB/8GB** → diffusion·value **동시 불가 = 순차 강제**. config의 **1e6(=~97h)은 비현실적 → 폐기.**
- **발사 = diffusion 200k + value 200k 순차 체인**(단일 background; `train.py --n_train_steps 200000` `;`
  `train_values.py --n_train_steps 200000`; 로그 `/tmp/p4_diff_full.log`·`/tmp/p4_val_full.log`; 각 ~19.4h
  합 ~38h; 세션 닫혀도 무인 완주). diffusion 발사 직후 정상(step200 loss 0.0544↓, GPU 93%).
- **★함정**: `n_steps_per_epoch=10000`이라 `--n_train_steps`는 10000 배수여야 epoch≥1(250→0 epoch=0 step만
  학습; 짧은 게이트엔 `--n_steps_per_epoch 200`도 함께). value 경로 실전검증 완료(loss 0.1959→0.0375, corr
  작동, ValueFunction fork-patch가 forward+backward+EMA 전 경로 OK). ★pkill 자기매칭 금지=PID로 kill.

## 2. 적대적 계획 검수 (critic (C) → 재검증 (B))
- critic 주장(검증됨): 데이터 내 최속 **2랩 완주 = cap-10 56.14s**, 37.3s보다 빠른 완주 **0/52**.
  → 구 floor(cap-15 37.3s 초과)는 2랩 완주 라벨이 데이터에 없는 영역.
- 내 1차소스 반박: **고속 *단일랩* 재료 빽빽**(cap-20 117랩@18.04s, cap-15 132랩@19.74s; 2랩 환산 ~36–40s).
  2랩 완주 0은 *고속 불가*가 아니라 stochastic 정책이 2랩 도는 중 충돌하기 때문. obs lap-blind + 평가 K=1
  MPC라 모델은 2.56s 윈도만 이어붙임 → 고속 윈도는 in-distribution. → floor는 OOD 불가능이 아니라 **불확실 stretch.**
- **★성공기준 재정의**: 승리=**baseline cap-5 107.16s 초과**(cap-10 56s가 이미 2×) / 목표=**≈56s**(cap-10
  복원) / stretch=**≈36–40s**(stitching+value-shaping 베팅). 보고서 서사 "expert급 복원"❌ → "충돌-위주
  데이터서 안전완주 복원 + stitching 탐색".
- **★진짜 linchpin = D3**: value가 speed 보상(corr+0.36), 충돌은 γ=0.99로 소멸(collision −10·lap +200 모두)
  → 안전-고속/충돌-고속 구분 못함. stretch의 실제 리스크이자 해결 레버. **v1=npz reward 유지**(계획 D4의
  축소 collision −2~−3은 D3 *악화*라 폐기, npz −10이 정답 — 문서만 정정). **D3 contingency**: v1 평가서
  충돌 징후 나오면 **value만 ~19h v2 재학습**(crash-aware reward), diffusion 재사용.

## 3. 적대적 구현 검수 (critic (B), 구현 code 버그 0)
- 구현된 P0~P4 코드(로더 역정규화·normalizer·모델·trainer·데이터 계약)는 **사양 충실·검증 버그 0**(critic +
  내 재검증). 내 1차소스 재확인: npz action∈[-1,1]·v_max per-tier(cap5→5..cap20→20) → 단일 역정규화 정당.
- **★[Critical] 평가설계 결함**: [007 §4]가 eval env `v_max=5.0` + `raw_to_norm(v_max=5.0)`로 명세 →
  SSOT S4(V_MAX=20)와 충돌. 그대로면 raw_to_norm clip으로 **차량 5 m/s 하드캡 = baseline급 = 목표 불가**
  (Diffuser는 cap-20에서 raw speed 최대 20까지 학습). **수정 = eval V_MAX=20 통일**(007 하단 정정 append).
- **추가발견(critic 놓침)**: [007 §2] "value sort 후 best-of-64"도 코드와 불일치 — 실제 [policies.py:36]
  `action[0,0]`(선별 없음). **value guidance gradient가 품질의 유일 레버** → D3 더 중요.
- **D5(실버그) 수정완료**: `config/f1tenth.py` `plan.discount 0.997→0.99`(value 저장 d0.99와 loadpath 일치;
  plan 블록=eval 전용이라 학습 무영향).
- 기타: S3=평가 K=1 확정(drift 없음, sim이라 latency 무관). S4=raw→env 재정규화 V_MAX=20 고정(P5).
  S5=코어 무변경 예외=ValueFunction fork-patch 1건. S1(패딩)·D2(서사)는 v2/문서 보류.
- normalizer 고정: §3 pickle 글루는 **현 diffusion run엔 미적용**(이미 실행 중) → **v1 eval은 frozen 데이터
  (`F1TENTH_DATA_DIR`+`F1TENTH_LIDAR_DOWNSAMPLE=128`) 재fit에 의존**(결정론적 동일). 추가 수집 금지.
- 잔여 한계: upstream diff 미수행 → "코어 무변경"은 grep(마커 부재)+학습 정상 수렴으로 *간접* 입증.

## 4. 데이터 다양성·커버리지 검증 (1차 실측, 그림)
- **deterministic 아님 확인**(crash_data 767ep): 중복(동일 복제) **0건**(전 tier), 출발점 동일하나 step30서
  pose 발산(std>0)·step5 action std(steer 0.22~0.27)=stochastic 작동, 종료/충돌 위치 수 m~20m 분산,
  ep 길이 10배+ 분산. 속도 다양성=tier 간, 라인 다양성=tier 내.
- **그림**: `_thinking/understand/crash_data_distribution-2.png`(6패널: 궤적/위치별속도/종료점/속도분포/
  지속시간/per-lap time). 결론: 트랙 전구간 커버 + 0~20 m/s 연속 스펙트럼 + 고속 단일랩(18–20s) 존재.
  데이터는 약한 고리 아님(단, stretch 보장은 D3/커버리지에 달림).

## 5. 이번 분기 변경 (working tree, git 미실행)
- 🟠 `config/f1tenth.py`(D5: plan.discount 0.99) / `plan_new/008`(검수반영·성공기준 재정의 append) /
  `analysis/007`(V_MAX=20 정정 append)
- 🟢 `_thinking/understand/001-diffuser-vs-value-architecture.md` / `_thinking/understand/crash_data_distribution-{1(폰트깨짐),2}.png`
- 메모리(~/.claude) P4/검수/재정의 갱신(git 무관).

## 6. 다음 (P5)
**유일 blocker(007 v_max)는 해소됨 → P5 착수 가능.** 순서(007 §5): device 파라미터화(policies.py:55→
self.device, arrays.py→DIFFUSER_DEVICE) → normalizer 처리(frozen 또는 pickle) → **plan_f1tenth.py 신설**
(value+scale=0.1, conditions{0:obs133} 1D, env_obs_to_cond=min-pool128, **raw_to_norm V_MAX=20**, K=1 MPC,
2랩 lap_time vs 107.16s) → verifier 1ep(action shape/normalize/2랩 정합). 학습 도는 동안 코드 작성 + CPU
plumbing 검증 가능(모델 없이 env/obs/action 왕복 확인).

## 7. 제약/규약 (불변)
★ git add/commit/push/pull 전부 사용자 지시 시만(자율 금지). 코어 무변경(ValueFunction fork-patch 예외 1건).
데이터=RL_project/runs/crash_data(gitignore, 동결). 모든 lap=2랩. 트랙=Oschersleben. GPU=run_in_background.

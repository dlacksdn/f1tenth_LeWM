# 003 — Diffuser offline RL 확정 계획 (critique + 병렬조사 + 교수님/사용자 입력 반영)

> 2026-06-13. plan_new/001(v1) → 002(critic 검토) → 본 문서(확정). 4-에이전트 병렬 조사
> 워크플로우(학습동역·speed-cap·centerline피처·Diffuser글루) 결과와 교수님 답변·사용자
> 데이터설계 통찰을 통합. 이 문서가 구현의 SSOT. 단 **데이터 소스/표현은 교수님 확인 2건 +
> P0/P1 게이트로 최종 확정**(아래 명시)되므로 파라미터화해 기술. append-only.

---

## 배경 (이 문서만 읽어도 되도록)

**과제**(AIE4003 개인): ~느린 성능의 policy로 주행 데이터를 모으고, **환경 추가 상호작용
없이(offline RL)** 더 빠른 주행 정책을 학습. 산출물 = behavior policy보다 빠른 lap time +
보고서. **Oschersleben 단일 트랙**(map_easy 건너뜀).

**모델**: Diffuser(Planning with Diffusion, Janner ICML 2022, 코드 `~/planning_with_diffusion`).
궤적 diffusion + value 함수를 offline 데이터로 학습 → 주행 시 현재 관측 조건으로 미래 궤적
생성하되 value gradient가 고-return(빠른 진행)으로 유도 → 첫 action 실행(MPC). LeWM(JEPA)에서
전환한 이유는 plan_new/001 배경 참조.

**핵심 원리 + 한계** (사용자와 합의): Diffuser는 데이터의 좋은 구간을 **재조합(stitching)** +
value guidance로 behavior policy를 넘는다. 단 **in-distribution 생성기라 데이터에 없는 속도는
외삽 못 함**. 따라서 **데이터에 빠른 재료가 실제로 들어있어야** 빠른 주행을 합성한다 — 데이터
설계가 성패를 결정한다.

**교수님 확정**: "60초 데이터→30초 가능, **정성적(qualitative) 평가**". baseline/목표 쌍 유연.
→ 성공 = 명확한 개선 + 좋은 분석(정확한 수치 임계 아님).

---

## 조사로 바뀐 핵심 사실 (plan_new/001 대비)

| # | 발견 | 출처 | 영향 |
|---|------|------|------|
| F1 | **저장 스냅샷 중 1랩 완주하는 것 없음**. 학습이 크래시(~3s)→고속완주(~18s)로 점프, 중간 완주단계 부재. reward에 속도항 없음→첫 완주가 이미 최속 | critique §4, training-dynamics | "초기 체크포인트=느린 완주 policy" 불가 (D2 폐기) |
| F2 | **GapFollower(휴리스틱 gap-following 컨트롤러)가 Oschersleben 30.36s 완주** (STRAIGHTS 9.0/CORNERS 6.0). RL 아님, 크래시 거의 없음 | training-dynamics | **moderate 완주 behavior policy를 깔끔히 제공** → 새 데이터 소스 |
| F3 | **centerline 피처는 기존 npz로 계산 불가** — npz에 pose 없음 + state가 clip·정규화됨. → **pose+raw속도 기록하며 재수집 필수** | centerline-features (Critical) | 데이터 재수집 전제. 기존 265-ep replay는 피처용으로 못 씀 |
| F4 | **action은 정규화 [-1,1]로 저장**(speed max 1.0=20m/s). speed-cap은 **raw 단계에서** 곱해야 함(SpeedCappedAgent wrapper) | speed-cap, 직접확인 | 데이터 파이프라인·캡 구현 |
| F5 | 기존 replay buffer = 265 ep(크래시 259, 완주 6), medium-replay 스타일. 풀스피드 데이터 있음 | 직접확인 | lidar 표현 시 "minimal-first" 데이터로 사용 가능(아래 대안경로) |
| F6 | Diffuser 글루: d4rl import가 여러 체인에 박혀 전수 차단 필요, normalizer 미저장(재로딩 재계산), plan_guided는 f1tenth 루프로 전면 교체, NullRenderer 스텁 필요. 코어는 무변경 | diffuser-glue | 적응 공수 ~4~6일 |

---

## 교수님 확인 대기 2건 (데이터 소스 확정의 전제)

1. **expert(깨끗한 완주 데이터, 예 DreamerV3 16.6s) 사용이 허용/적절한가?**
2. **단일 지정 behavior policy 데이터여야 하나, 아니면 여러 소스 혼합(replay/GapFollower/체크포인트)도 되나?** (과제 문구 "policy로 수집"이 단일 policy 의도일 수 있음)

→ 이 답에 따라 아래 D2가 갈림. 그 전까지는 **GapFollower 단독 경로**가 가장 안전한 기본값.

---

## 설계 결정

### D1. 관측 표현 — centerline 피처 1순위 (재수집), lidar 백업
critique 권고대로 raw lidar 대신 **centerline 상대 저차원 피처**. 이유: 미래 lidar 생성이 진짜
리스크인데 피처는 생성 쉽고·Markov·progress 정렬·자기교차 모호성 해소.

**피처(~9~11차원)** (centerline-features 권고):
`[e_y(횡오차), sin(e_ψ), cos(e_ψ), v_long, v_lat, yaw_rate, κ(s+L1), κ(s+L2), κ(s+L3), (progress=s/L)]`
- e_y = (pos − 최근접 중심선점)·왼쪽법선. e_ψ = wrap(yaw − 접선각), **sin/cos 분해**(경계 불연속 제거).
- 곡률 κ는 **lookahead arclength heading-rate 또는 spline**(raw 접선 미분은 노이즈 과다 — 금지).
- 최근접 인덱스는 **env의 windowed 순차 추적 재현**(전역 argmin은 자기교차에서 가지 점프).
- ⚠️ **재수집 필수**(F3): pose + raw 속도 + trackname 기록. raw lidar도 함께 저장해 표현 교체 대비.
- **대안(minimal-first)**: 재수집이 부담이면 기존 265-ep replay + **lidar 다운샘플(~64)** 로 파이프라인
  먼저 검증 → 안 되면 피처로 재수집. 사용자 incremental 선호 반영.

### D2. 데이터 소스 — 완주 속도 스펙트럼 (교수님 답변으로 확정, 기본값=GapFollower)
원리(합의): 데이터에 **빠른 재료 + 완주 궤적**이 있어야 stitching이 빠른 주행을 합성. F1으로
"느린 완주 RL policy"는 못 만듦 → 대안:

- **기본값(가장 안전, 교수님 답 전 진행 가능): GapFollower + speed-cap 스펙트럼** (F2,F4).
  GapFollower가 30.36s로 완주 → SpeedCappedAgent로 {×1.0≈30s, ×0.7, ×0.5, ×0.3} rollout →
  **완주하는 속도 스펙트럼**. baseline=느린 캡, 데이터엔 빠른 캡(×1.0) 포함 → 향상 재료 확보.
  - 한계: GapFollower 단일 라인이라 **속도 다양성은 있으나 라인 다양성 부족**. 보강:
    GapFollower 파라미터 변형(STRAIGHTS/CORNERS 임계) + action noise로 라인 다양성 추가.
- **확장(교수님 허용 시)**: DreamerV3 expert(16.6s, 다른 빠른 라인) + 체크포인트 rollout +
  기존 replay 혼합 → 라인 다양성↑ → 향상 폭↑. 사용자 통찰(replay의 풀스피드+완주6개 활용).
- **사용자 incremental 원칙**: expert 없이(GapFollower/replay) 먼저 → 안 되면 expert 추가.

### D3. speed-cap 구현 (speed-cap 조사)
- **SpeedCappedAgent wrapper**: 정책 출력(정규화 action)을 NormalizeActions 규약대로 **raw로
  역정규화 → raw speed에 κ 곱 → 재정규화**해 simulate에 전달. **raw 단계 곱 필수**(정규화 곱은
  offset 때문에 비비례 — 금지). steer는 무수정 통과. env 무변경(차체 물리 불변 준수).
- κ ∈ {1.0, 0.7, 0.5, 0.3} (필요시 0.2). 각 κ의 실제 lap time은 **rollout로 측정**(log_lap_time_s
  합, log_completed) — baseline 숫자 확정 게이트.
- ⚠️ 리스크: 저속에서 steer 무스케일이라 **코너 오버스티어로 완주 실패** 가능. 저속캡이 완주
  못 하면 κ 하한 조정 또는 그 캡 제외.

### D4. value 대상 reward (critique §5)
- **progress(dense, `log_reward_progress`) + 소형 collision penalty(-2~-3으로 축소)**, **lap
  보너스(+100) 제외**(극희소→value 분산 폭증). **normed value([-1,1], ValueDataset normed=True)**.
- ⚠️ 속도캡 데이터의 value 함정: 데이터 최대 progress가 낮으면 guidance 천장도 낮음 → 빠른 캡(×1.0)
  데이터 포함이 value 학습에도 필수.

### D5. Diffuser 적응 (diffuser-glue, 코어 무변경)
구현 touch-point (file:line은 analysis/005에 상세 기록 예정):
1. **데이터 로더 어댑터 신설**: `diffuser/datasets/f1tenth.py` — `load_f1tenth_environment`(더미 env:
   .seed no-op/._max_episode_steps/.name) + `f1tenth_sequence_dataset`(에피소드별
   observations/actions/rewards/terminals[/timeouts] yield, d4rl.py 시그니처 동일).
2. **d4rl import 전수 차단**: d4rl.py:23-26 `import d4rl`를 try/except graceful degrade
   (datasets/__init__.py:2, preprocessing.py:7, rendering.py:15 체인 전부 영향). **첫 게이트(P0)**.
3. **NullRenderer 스텁** + config renderer 교체 + train config `sample_freq=0`(렌더 경로 우회).
   rendering.py의 mujoco_py import가 `utils` 전체 import를 죽일 수 있으니 분리.
4. **normalizer 영속화 보강**: 현재 미저장·eval시 데이터 재로딩 재계산(serialization.py:47).
   → 데이터 경로 고정+결정적 로딩 보장, 권장은 Trainer.save에 normalizer pickle 덤프 추가.
5. **평가 루프 재작성**: `scripts/plan_f1tenth.py` — 로딩부 재사용, env 루프(plan_guided.py:76-118)를
   f1tenth sim 루프로 교체. `conditions={0: obs}`만 맞추면 GuidedPolicy 그대로 동작. device='cuda:0'
   하드코딩(policies.py:55) 파라미터화.
6. **buffer.py:12 `np.int`→`np.int64`** (numpy≥1.24 즉시 에러 방지). termination_penalty 쓰면
   timeouts 키 필수(buffer.py:78-80 assert) → 안 쓰려면 config에서 None.
7. **config 신설**: `config/f1tenth.py` (loader/normalizer/renderer 문자열, horizon/max_path_length).

---

## 향상 폭 현실 추정 (critique §3 + 교수님 정성평가)
- 데이터에 ×1.0(≈30s, GapFollower) + (허용 시)expert(17s) 라인이 포함되면, Diffuser+guidance는
  데이터 최속 근처~약간 개선(라인·속도프로파일 합성)을 낼 가능성. baseline(느린 캡, 예 ×0.3) 대비
  **개선은 명확히 달성 가능**. 정성평가라 정확한 수치 임계 부담 없음.
- 데이터를 균일 느린 단일캡으로만 모으면 개선 미미 → **빠른 재료 포함이 필수**(D2).

---

## Phase 분해 (게이트 포함, 수정판)

| # | 작업 | 게이트 |
|---|------|--------|
| P0 | Diffuser 핵심+글루 py3.8(torch2.4) smoke: d4rl 전수 차단 + 더미 로더로 train.py **1-step 학습** + NullRenderer | import 통과 + 1-step loss 산출 |
| P1 | **데이터 소스 확정**(교수님 답+) + behavior policy 준비(GapFollower±expert) + **각 κ baseline lap time rollout 실측** | 완주 확인 + baseline 숫자 |
| P2 | **재수집**(pose+raw속도+trackname 기록): κ 스펙트럼 rollout, raw lidar+state+pose 저장 | 궤적 무결성, 분포 리포트 |
| P3 | 표현 결정(centerline 피처 1순위/대안 lidar) + f1tenth 로더 + online 피처 모듈 + normalizer 저장 | SequenceDataset 통과, 윈도우 검산 |
| P4 | 궤적 diffusion + value 함수 학습 | loss 수렴 + **생성 궤적 품질 점검**(미래 피처 매끄러움) |
| P5 | plan_f1tenth.py: GuidedPolicy↔f1tenth 루프(normalizer 로드+online 피처+MPC) | 1 ep 완주 시도 |
| P6 | 평가: baseline(느린 캡) 대비 lap time, 2랩 완주, Diffuser가 더 빠른지 | **baseline보다 빠름(정성)** |

순서 주의: 표현(centerline)을 쓰면 P2 재수집이 P3보다 앞. minimal-first(lidar+기존replay)면 P2 생략 가능.

---

## 리스크 Top + 완화
1. **데이터 빠른재료/라인다양성 부족**(향상 천장) → GapFollower 파라미터변형+noise, 허용 시 expert/replay 혼합.
2. **저속캡 완주 실패**(D3 오버스티어) → κ 하한 조정/제외, rollout 실측 게이트.
3. **centerline 피처 재수집·online 계산 정확도**(자기교차 인덱스, 곡률 노이즈) → env windowed 추적 재현, spline/lookahead 곡률, P3 검산.
4. **Diffuser 글루 d4rl import 폭탄 + normalizer 비영속** → P0 전수 차단 smoke, normalizer 덤프.
5. **완주 데이터 희소**(replay 6/265) → GapFollower로 완주 다수 확보(F2).

## 다음 단계 (인수인계용)
1. 교수님 답 2건 확보 → D2 데이터 소스 확정.
2. P0(글루 smoke)는 교수님 답 무관하게 선행 가능 — 바로 착수 가능.
3. diffuser-glue 상세 touch-point(file:line)는 analysis/005로 분리 기록 예정.
4. 프로젝트 폴더 `f1tenth_LeWM`→`f1tenth_planning_with_diffusion` 리네임 예정(사용자). clone=`~/planning_with_diffusion`.
5. 분기마다 _thinking 문서 + commit + push.
6. (선택) 본 plan_new/003을 적대적 검토 워크플로우로 한 번 더 돌려 확정.

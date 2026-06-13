# 005 — Phase 2 확정 설계서 (003 + 004 critique 전면 반영)

> 2026-06-13. plan/003(설계 결정 4개 v1)에 대한 critic 검토 plan/004를 전면 수용한 **확정판**.
> 이 문서가 Phase 3 구현의 SSOT다 — 구현 중 이 문서와 어긋나는 발견이 나오면 새 문서로
> 기록하고 여기를 참조한다. append-only.
>
> critique의 Critical 1건은 본 세션에서 교차 검증 완료: py3.10 venv에 gym/numba 없음(import
> 실측), vendored f110_env.py:394-410이 **활성** `import gym`+`import pyglet` 수행 — 즉
> "평가 단계에서 py3.10 프로세스에 시뮬레이터가 없다"는 지적은 사실.

---

## 배경 (이 문서만 읽어도 되도록 요약)

**프로젝트**: f1tenth 자율주행 시뮬레이터에서 LeWM(LeWorldModel — 이미지+action 시퀀스로
latent dynamics를 학습하고 MPC/CEM으로 planning하는 JEPA 계열 world model, 공식 코드
`~/le-wm`, 무수정 사용이 목표)을 offline 데이터셋으로 학습시켜 주행까지 시키는 개인 프로젝트.
데이터는 직전 DreamerV3 프로젝트(`~/f1tenth_RL_project`)의 학습된 policy snapshot들로
rollout해서 만든다. **절대 제약: f1tenth 차체 물리값 무변경.**

**전사(이미 끝난 것)**: 환경 구축 + 공식 데이터로 학습 파이프라인 smoke 통과
(implementation/001), 데이터 파일 규약·로더·planning 스택 소스 분석(analysis/002),
모델 크기는 정본 유지 확정(analysis/001), 렌더 프로토타입 검증(spikes/), 설계 v1(plan/003)과
critic 검토(plan/004).

**이 문서**: 검토 반영 후의 최종 설계. 4개 결정 + 평가 런타임(신규) + Phase 3 작업 분해.

---

## 결정① 관측 모달리티: ego-centric top-down 래스터 — 확정

- 224×224 RGB uint8, ego-centric(차량 중심), heading-up 고정, **시야 폭 22.4m**(0.1 m/px)
  - 시야를 좁히지 않는 이유(P-1): goal 도달 비용이 이미지의 "장소 식별력"에 의존하는데,
    좁은 복도 뷰는 직선 구간이 서로 닮아(place aliasing) goal 매칭이 무너질 위험. 22.4m면
    맵(30×33m)의 절반가량이 보여 위치 시그니처가 강함
- **벽 두께 스펙 = 최종 224px 이미지에서 ≥2px** (원본 해상도 dilation을 역산하면 ~10px=0.2m).
  스파이크 이미지의 벽이 hairline(~1px)이라는 실측 지적 반영
- 게이트: 육안 검증 + **벽 픽셀 두께/비율 자동 검사**
- 맵 밖 = 검정(점유 의미), 차량 = 중앙 빨강 삼각형
- **숨은 전제(기록)**: planning 시 모델 입력은 현재 프레임 1장 — 이미지에 속도/요레이트가
  없다. f1tenth action의 speed가 절대 목표속도(PID 추종)라 대부분 완화되지만, 가감속
  과도구간에서 위치 예측 오차 가능. → M0 진단 항목 + 실패 시 1순위 처방 = **속도 글리프**
  (차량 마커 앞에 속도 비례 선분 렌더). npz→재렌더 구조라 수집 재실행 없이 적용 가능
- 기각 대안: (b) encoder lidar 변형 — train/eval/swm 연쇄 수정 필요(코드 확인), 렌더 성공으로
  정당성 상실

## 결정② 데이터셋: map_easy3, 3-소스 믹스, 2단 수집 — 확정

### 규모·수집 (2단화 — critique §6 과잉 지적 반영)
- **1차 500 ep(~200k transition) → 학습+M0-lite 진단 한 바퀴 → 렌더 스펙 확정 → 잔여 2,000 ep**
- 에피소드: **가변 길이** (충돌 등 terminated 조기 종료 포함, cap 400 step(8초), 20 step 미만
  폐기). "충돌 직전 상태"가 noise 소스의 존재 이유이므로 조기 종료 에피소드도 데이터에 포함
- 파일럿 50 ep 게이트: ep_len 히스토그램, 폐기율(<20 step, 목표 <10%), noise 소스 충돌
  종료율 10~25%, 용량/렌더 시간 실측

### 정책 믹스
| 소스 | 비율 | 내용 |
|---|---|---|
| diversity bin policy | 60% | distinct lap 10개(6.1~18.4s) 균등 — ★count 확정: 6.1 포함 10개 |
| best policy (6.1s) | 20% | expert 영역 + M1 비교용 |
| best + action noise | 20% | **에피소드별 ε 샘플**: σ_steer~U(0.03,0.12)rad, σ_speed~U(0.5,2.0)m/s |

### 시작 pose (m-4 안전 규칙 반영)
- s-위치: **균등 격자 + 지터** (소스·bin별 스트래티파이) — 전 구간 커버리지 보장
- 횡 오프셋 ≤ min(0.6m, clearance(s)−0.45m) — clearance는 점유맵 distance transform으로
  사전 1회 계산해 centerline csv에 컬럼 추가 (critique 실측: 중앙값 1.37m, p5 0.94m)
- heading 섭동 ±15°, 스폰 직후 collision_raw 확인해 충돌이면 재추첨
  (wrapper의 ignore_first_collision이 가릴 수 있음 주의 — f1tenth_env.py:321-323)

### action 기록 규약 (M-2 — silent failure 차단)
- **h5의 action = env에 실제 적용된 raw 물리 스케일 (steer[rad], speed[m/s]), noise·clip
  반영 후 값**
- 게이트: 기록 action을 동일 시작 pose에서 재주입 → **동일 궤적 재현** 확인(결정적 sim).
  action 무결성과 pose 무결성을 동시에 검증하는 가장 싼 테스트

### h5 스키마 (analysis/002 규약 + blosc)
- 필수: pixels(N,224,224,3)u8 / action(N,2)f32 / proprio(N,5)f32 / ep_idx, step_idx,
  ep_len, ep_offset. 에피소드 마지막 row action=NaN
- 진단(로더 무시): pose(N,3), arclen_s, lap 카운트, 소스 policy 라벨
- h5py 직접 작성 + hdf5plugin.Blosc (공식 Writer는 무압축이라 미사용, 코드는 본뜸)
- 게이트: `swm.data.load_dataset` 통과 + 윈도우 수 검산
- 수집(py3.8, npz 중간 저장) → 렌더+변환(py3.10) 분리 유지 — 렌더 스펙 반복 수정 가능

## 결정③ 평가: receding subgoal-chasing + 계단 milestone — 확정 (대폭 보강)

### 평가 런타임 (C-1 — 신규 결정)
- **Phase 3-0 스파이크 (최우선, 3-1a와 병행)**: py3.10 venv에 f110_gym 구동 시도 —
  `pip install gym==0.18.0 --no-deps` + pyglet 1.5.x + numba + f110_gym editable →
  wrapper import → 1 ep rollout. 게이트 = 같은 seed/pose에서 py3.8과 lidar/state 일치
- 실패 시 fallback (사전 확정): ① py3.8 env 서버 ↔ py3.10 planner 소켓/파이프 브리지
  (인터페이스 = pose/obs/action만이라 작음) ② f110_gym을 본 프로젝트에 포팅(gym→gymnasium,
  pyglet import 제거 — **물리 코드 무변경이므로 사용자 제약 위반 아님**)

### swm 인터페이스 명세 (M-1, m-1, m-2 — 첫 실행 즉사 방지)
- **VecEnv 셔임** 필수: `num_envs=1`, `single_action_space=Box(low,high,(2,))`,
  `action_space=Box(low,high,(1,2))` — WorldModelPolicy가 vector-env 형태를 전제
  (policy.py:362,427,434 / cem.py:64가 단일 Box(2,)면 action_dim=1로 오계산)
- infos dict 명세: `pixels(1,1,224,224,3)u8 HWC`, `proprio(1,1,5)`, `goal(1,1,224,224,3)`,
  **`action(1,1,2)`(직전 실행 action — jepa.py:145의 기본값 없는 pop 때문에 존재 필수)**,
  goal 키는 매 step 재주입, **reset마다 `infos['_needs_flush']=[True]`**
- eval yaml: `dataset.keys_to_cache=[action, proprio]` (scaler fit 대상)

### CEM 클램핑 — M1부터 기본 적용 (리스크 3 반영)
- CEM 샘플을 z-score 공간 ±2σ로 clip하는 외부 콜백 (정본 무수정). 이유: unbounded 샘플은
  env clip으로 실행은 막아도 **cost 평가가 OOD action으로 이뤄지는** 환각을 못 막음

### Milestone
- **M0-lite (M0 decoder 대체, 선행)**: 데이터셋 ~10k 프레임 임베딩 인덱스 → open-loop
  rollout 예측 임베딩의 NN-retrieval 시각화. 학습 0, 트랙 구조 상상 + place aliasing 동시 진단.
  decoder 학습은 M0-lite가 애매할 때만
- **M1 (goal-reaching)**: 고정 시작 pose에서 +25 env step 앞 goal 도달률
  - 판정: pose 거리 ≤1.0m **AND |Δs| ≤2.0m** (자기교차 트랙 오판 방지), budget 50 env step
    내 최초 충족
  - 레퍼런스: **중간 bin(8~9s) lap** (best의 공격적 라인은 비교용 상한)
  - 베이스라인: random(하한) + **레퍼런스 action open-loop 재생(상한 — 100% 미달 시 모델이
    아니라 하니스/렌더/정규화 버그)**
  - 실패 시 진단 순서 고정: ①하니스(상한 베이스라인) → ②aliasing(M0-lite) → ③속도 비관측
    (결정① 글리프) → ④시간해상도(결정④ v2)
- **M2 (완주)**: goal = **s-indexed 합성 렌더** — render(centerline pose at arclen_s+Δs),
  Δs≈2.5~4m, arclen_s는 env info 제공(f1tenth_env.py:457). 레퍼런스 lap 시간 동기화 문제를
  구조적으로 제거(P-5/M-4), goal 생성 비용 0. 지표 = 완주 여부, lap time (참고: Dreamer 대비)

## 결정④ 시간 해상도: frameskip 5 유지 — 확정

- env step 0.02s → 모델 step 0.1s, lookahead 0.5s(=4m@8m/s). 정본 검증값 유지
- frameskip은 로더 파라미터라 **h5 재작성 없이 변경 가능**(소스 확인). 단 10 전환 시 부수효과:
  span 20→40(짧은 에피소드 추가 탈락), action_encoder input 10→20(**모델 재학습**),
  PlanConfig.action_block=10 동행 변경
- 고속 코너 실패 시 frameskip보다 **속도 비관측(결정① 숨은 전제)을 먼저 의심** — 진단 순서 참조

## 고정 사항 (재확인)

- 모델: 정본 config(18.04M 실측), 10 epoch, batch 사전 스윕. λ=0.09
- train/val: window-level 0.9 split 수용 (낙관 편향 인지)

## Phase 3 작업 분해 (확정)

| # | 작업 | 게이트 |
|---|---|---|
| **3-0** | **py3.10 f110_gym 구동 스파이크 (3-1a 병행)** | 1 ep rollout + py3.8와 obs 일치. 실패 시 fallback ①브리지/②포팅 결정 |
| 3-1a | 수집기(py3.8): snapshot 로드 + pose/arclen 기록 + npz, 파일럿 50 ep | action 재주입 궤적 재현, 물리값 무변경, ep_len/폐기율/충돌률 리포트 |
| 3-1b | 렌더러(py3.10): npz→이미지 (벽 두께 ≥2px 스펙) | 연속 프레임 육안 + 벽 두께 자동 검사 |
| 3-1c | h5 변환기: blosc + NaN 규약 + 인덱스 | load_dataset 통과 + 윈도우 수 검산 |
| 3-1d | **1차 수집 500 ep** (믹스/시작 pose 정책) | 용량·분포 리포트 |
| 3-2 | data/f1tenth.yaml + batch 스윕 | smoke 1 epoch |
| 3-3 | 학습 10 epoch → **M0-lite** | loss 수렴 + retrieval 진단 통과 |
| 3-3' | 렌더 스펙 확정 → 잔여 2,000 ep 수집 → 재학습 | — |
| 3-4 | 평가: VecEnv 셔임 + 루프 + M1 → M2 | M1 상한 베이스라인 100%, M1>random, M2 완주 |

## 다음 단계 (인수인계용)

1. 이 확정 설계로 **Phase 3-0 + 3-1a 병행 시작** (둘 다 반나절급, 의존성 없음)
2. 실행 환경: env_setting/004 (venv 경로·STABLEWM_HOME·함정 목록)
3. 모든 분기: _thinking 문서(analysis/implementation/plan 라우팅) + commit + push
4. critique에서 미확인으로 남긴 것 중 구현에서 자연 해소될 항목: blosc 압축률(3-1c 실측),
   렌더 처리 시간(3-1b 실측), gym 0.18 py3.10 설치 가능성(3-0이 답)

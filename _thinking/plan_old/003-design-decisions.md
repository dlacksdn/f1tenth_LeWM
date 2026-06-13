# 003 — Phase 2 설계서: 설계 결정 4개 (v1)

> 2026-06-13. 계획 v2(plan/002)의 Phase 2 산출물. Phase 1 전체 결과(implementation/001,
> analysis/001~002)를 입력으로 4개 설계 결정을 확정 제안한다. critic 검토 후 Phase 3 진입.
> 근거 문서: [스키마/로더] analysis/002 §1, [planning 경로] analysis/002 §2-3,
> [모델 크기] analysis/001 §1, [논문 스케일/리스크] implementation/001 §6. append-only.

---

## 배경 (이 문서만 읽어도 되도록 요약)

**프로젝트**: f1tenth 자율주행 시뮬레이터에서 LeWM(LeWorldModel — JEPA 계열 world model,
이미지+action 시퀀스로 latent dynamics를 학습하고 MPC로 planning하는 모델, 공식 코드
`~/le-wm`)을 offline 학습시켜 주행까지 시키는 개인 프로젝트. 데이터는 직전 DreamerV3
프로젝트(`~/f1tenth_RL_project`)에서 학습해둔 policy들로 rollout해서 만든다.

**지금까지 확인된 핵심 사실** (상세는 각 문서):
- LeWM 입력 = (224×224 RGB 이미지, action) 시퀀스의 HDF5 파일. 우리 f1tenth 관측은 lidar
  벡터라 **이미지를 새로 만들어야 함** → 맵+차량위치(pose)로 위에서 내려다본(top-down)
  이미지를 합성하는 방식이 프로토타입으로 검증됨 (implementation/001 §5)
- 공식 코드는 smoke test 통과 (공식 tworoom 데이터셋으로 학습 파이프라인 작동 확인)
- 데이터 파일 규약(스키마), 로더의 윈도우 생성 방식, planning 시 정규화 경로까지 소스 레벨로
  분석 완료 (analysis/002)
- 모델 크기는 정본 그대로 유지(18M, 줄이면 성능 급락하는 ablation 있음 — analysis/001)

**이 문서의 역할**: 구현(Phase 3) 전에 정해야 하는 설계 결정 4개 — ①관측 이미지를 어떻게
만들지 ②데이터셋을 어떻게 구성할지 ③학습된 모델을 어떻게 평가할지 ④시간 해상도를 어떻게
잡을지 — 를 근거와 함께 확정 제안한다.

---

## 0. 목표 재확인

f1tenth 시뮬레이터에서 (이미지, action) offline 데이터셋을 만들어 **정본 LeWM을 무수정으로
학습**시키고, 학습된 world model + swm planning 스택(MPC/CEM)으로 **goal-reaching → 완주**까지
평가한다. 절대 제약: f1tenth 차체 물리값 무변경.

---

## 결정① 관측 모달리티: **(a) ego-centric top-down 래스터 이미지** — 확정 제안

### 선택지 비교
| | (a) 래스터 이미지 | (b) encoder lidar 변형 |
|---|---|---|
| 정본 수정 | **0줄** | train/eval/swm 파이프 연쇄 수정 (HF ViT API 결합 jepa.py:37-38, 'pixels' 키 고정 train.py:59·eval.py:61-64, goal 인코딩 포함) |
| 실행 가능성 | **스파이크로 검증됨** (SPIKE_OK, numpy/PIL, headless) | 미검증, 논문 "from Pixels" 정체성 이탈 |
| 비용 | 렌더 파이프라인 + 저장 용량 | 코드 위험 + 비교 가능성 상실 |

### (a) 확정 + 렌더 스펙 v1
- 224×224 RGB uint8, **ego-centric, heading-up 고정** (스파이크 방식)
- 시야 폭 22.4m (0.1 m/px) — lookahead 0.5s × 최고속 ~8m/s = 4m 이동을 충분히 포함
- 시인성 보강(스파이크 잔여 항목): 벽 라인 dilation(원본 해상도에서 3px) 후 다운샘플,
  맵 경계 밖 = 검정(occupied 의미), 차량 = 중앙 빨강 삼각형
- 입력 데이터: rollout 시 raw env에서 (poses_x, poses_y, poses_theta) 병행 기록 (npz에 pose
  없음이 실측 확인됨 — implementation/001 §5)
- **게이트**: Phase 3-1 첫 에피소드 렌더를 눈으로 검증(연속 프레임 일관성, 회전 방향 정합)
  후 본 수집

### 기각 사유 기록
(b)안(LeWM encoder를 lidar 1D용으로 교체)은 단순 인코더 교체가 아니라 train/eval/swm
파이프라인 전반의 연쇄 수정임이 코드 분석으로 확인됐다(plan/001 검토 의견 M-4 = 위 표의
"정본 수정" 행). 렌더 프로토타입이 성공한 이상 이 비용을 감수할 이유가 없다.

---

## 결정② 데이터셋 구성: **map_easy3 단일, 3-소스 믹스, ~1M transition 목표**

### 트랙: map_easy3 단일 (v1). Oschersleben 확장은 v2
### 규모: **2,500 에피소드 × ~400 step ≈ 1M transition** (paper 최소 사례 TwoRoom 920k과 동급)
- env step 0.02s 기준 에피소드 400 step = 8초 ≈ best lap 1바퀴, 느린 policy는 부분 lap
- 에피소드 길이 하한: **20 step 미만은 로더가 통째로 버림**(span=20, analysis/002 §1) —
  충돌 즉사 에피소드도 20 step 이상 되도록 최소 길이 보장 or 폐기 규칙 명시
- 수집 시간/용량은 Phase 3-1 파일럿(50 ep)에서 실측 후 필요시 규모 조정 (축소는 쉬움)

### 정책 믹스 (SIGReg 다양성 리스크 대응 — implementation/001 §6)
| 소스 | 비율 | 근거 |
|---|---|---|
| diversity bin policy 9개 (lap 6.1~18.4s) | 60% (bin당 균등) | 속도·주행선 다양성의 주 공급원 |
| best policy (6.1s) | 20% | expert 영역 커버 + 레퍼런스 lap 생성원 |
| best policy + action noise (ε-Gaussian) | 20% | 벽 근접·회복 등 비정상 상태 커버리지 |
- 논문 요건 "pseudo-expert or exploratory, 커버리지가 중요" 충족 설계
- **시작 pose 다양화**: centerline 위 랜덤 s-위치 + 소량 횡방향/방향 섭동으로 reset
  (전 구간 커버리지 확보, 물리값 무변경 — 시작 pose는 물리 파라미터 아님)

### h5 스키마 (analysis/002 §1 규약 그대로)
- 필수: `pixels (N,224,224,3) uint8` / `action (N,2) f32` / `proprio (N,5) f32`(=state)
  / `ep_idx, step_idx, ep_len, ep_offset`
- 규약: 에피소드 마지막 row action = NaN
- 진단(로더 무시): `pose (N,3)`, `log_lap_count_arc`, `log_completed`, 소스 policy 라벨
- **압축: h5py 직접 작성 + `hdf5plugin.Blosc`** (공식 Writer는 무압축 → 150GB급. blosc로
  ~14GB 예상). 작성 후 `swm.data.load_dataset` 통과 = 게이트
- keys_to_load = [pixels, action, proprio] (tworoom과 동일 패턴)

### 수집 파이프라인 경계 (env_setting/004)
- py3.8(RL venv): policy 구동(lidar+state obs) + pose 기록 + **중간 npz 저장** (이미지 렌더는
  pose에서 후처리 가능하므로 수집 루프 밖)
- py3.10(LeWM venv): npz → 렌더 → blosc h5 변환 → 로더 검증
- 장점: 수집 재실행 없이 렌더 스펙 반복 수정 가능 (pose만 있으면 재렌더 가능)

---

## 결정③ 평가 설계: **(③-a) receding subgoal-chasing, milestone 2단 계단**

### 구조 (analysis/002 §3의 재사용 경계 그대로)
- 재사용: `WorldModelPolicy` + `CEMSolver` + process(StandardScaler)/transform(ImageNet) 스택
- 신규: f1tenth 평가 루프 — env reset → infos(pixels 실시간 렌더, proprio, goal) 구성 →
  `policy.get_action(infos)` → env.step → subgoal 갱신 → 판정
- goal 공급: **레퍼런스 lap**(best policy 에피소드 1개의 프레임 시퀀스)에서 현재 진행도(s) 기준
  +25 env step 앞 프레임을 goal로. 재계획 주기(receding 5×block 5=25 step)와 일치
- cost = 정본 `jepa.get_cost`(goal 임베딩 MSE) 무수정

### Milestone 계단
1. **M1 (goal-reaching, 정본 평가 방식 모사)**: 고정 시작 pose에서 +25 step 앞 subgoal 도달률.
   판정 = pose 거리(예: 1.0m 이내, 진단 컬럼 기반 — 임베딩 아닌 물리 거리). 베이스라인 = random policy
2. **M2 (완주)**: subgoal 체인으로 1 lap 완주 여부 + lap time. (참고 비교: Dreamer lap time)
- M1 실패 시 M2 진입 안 함 — 원인 분석(데이터/렌더/시간해상도) 후 재시도
- 사전 진단(M0): **사후 decoder open-loop rollout 시각화**(논문 v3 Fig.7 기법, analysis/001 §2)
  — MPC 전에 world model이 트랙 구조를 상상하는지 정성 확인

### 주의점 명시
- CEM unbounded 샘플 → env wrapper clip 의존 (analysis/002 §2). M1 결과 보고 필요시
  solver 클램핑 추가
- infos의 goal 키는 step마다 재주입 (world.py:541 패턴)
- ③-b(custom cost head)는 M1/M2 실패 시 fallback으로 보존

---

## 결정④ 시간 해상도: **정본 frameskip=5 유지 (v1)**

- env step 0.02s → 모델 step 0.1s, context 3프레임 = 0.3s, horizon 5 = **lookahead 0.5s**
- 0.5s × 8m/s = 4m 전방 — map_easy3 코너 스케일에서 1차 시도 값으로 타당, **정본 충실 원칙**
  (논문 검증값, 비교 가능성)
- planning 시 context는 1프레임(PlanConfig.history_len=1, 실측 — analysis/002 §2) — 정본 그대로
- subgoal 간격 = 25 env step = lookahead와 일치 (결정③과 정합)
- M1/M2에서 고속 코너 실패 패턴이 보이면 frameskip 10(lookahead 1s) 실험은 v2로 —
  단 frameskip 변경 = 데이터 재사용 가능(로더 파라미터일 뿐, h5 재작성 불필요. 매 step 저장 덕분)

---

## 고정 사항 (결정 아님, 확정 기록)

- 모델: 정본 config 그대로 (18.04M 실측, 축소 금지 — analysis/001 §1). embed 192, ViT-tiny,
  predictor 공개 config, λ=0.09
- 학습: 10 epoch (논문), batch는 사전 스윕으로 최대값 (smoke: 16에서 8GB 내 동작 확인)
- action_dim=2 → action_encoder input = 5×2=10 (train.py:68 자동 배선)
- train/val: 코드 내장 window-level 0.9 split 수용 (val loss 낙관 편향 인지하고 게이트로는
  loss 추이 + M0 시각 진단 병용)

## Phase 3 작업 분해 (게이트 포함)

| # | 작업 | 게이트 |
|---|---|---|
| 3-1a | 수집기(py3.8): policy 로드 + pose 병행 기록 + npz, 파일럿 50 ep | pose/액션 무결성, 물리값 무변경 확인 |
| 3-1b | 렌더러(py3.10): npz→이미지 (스파이크 모듈화 + dilation) | 연속 프레임 눈 검증 |
| 3-1c | h5 변환기: blosc + NaN 규약 + 인덱스 테이블 | `load_dataset` 통과 + 윈도우 수 검산 |
| 3-1d | 본 수집 2,500 ep (믹스/시작 pose 정책 적용) | 용량/분포 리포트 |
| 3-2 | data/f1tenth.yaml + batch 스윕 | smoke 1 epoch 통과 |
| 3-3 | 본 학습 10 epoch | loss 수렴 + M0 decoder 진단 |
| 3-4 | 평가 루프 + M1 → M2 | M1 도달률 > random, M2 완주 |

## 미결/critic 검토 요청 포인트

1. 렌더 시야 폭 22.4m vs 더 좁게(예: 12m, 0.054m/px) — 벽 시인성과 lookahead 커버의 트레이드오프
2. 믹스 비율(60/20/20)과 noise ε 크기의 근거 약함 — 더 나은 휴리스틱?
3. M1 판정 거리 1.0m의 적정성 (트랙 폭 대비)
4. 에피소드 400 step 고정 vs 가변(충돌 시 조기 종료 허용) — 분포 영향
5. 레퍼런스 lap 1개 의존 — subgoal 체인이 단일 주행선에 과도하게 고정되는 위험

## 다음 단계 (이 문서 이후의 진행 — 다른 세션 인수인계용)

1. **critic 검토**: 이 설계서를 별도 세션에서 비판적으로 검토 (검토 결과는 plan/004로 저장
   예정). 검토자는 위 "검토 요청 포인트" 5개 + 사실 검증(인용 file:line)을 다룬다
2. **검토 반영 → 설계 확정** (필요시 plan/005 수정판)
3. **Phase 3-1a부터 구현 시작**: 위 "Phase 3 작업 분해" 표 순서대로. 첫 작업 = py3.8
   (f1tenth_RL_project/.venv)에서 policy snapshot 로드 + pose 병행 기록 rollout 파일럿 50 ep
4. 실행 환경/커맨드/함정은 env_setting/004 참조 (STABLEWM_HOME, hdf5plugin 등)
5. 모든 분기마다: _thinking에 문서(분석=analysis/, 구현=implementation/, 계획=plan/) + commit + push

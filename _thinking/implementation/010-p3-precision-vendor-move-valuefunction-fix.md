# 010 — P3 정밀화(역정규화·min-pool·SafeLimits·horizon) + vendor 통합 이동 + ValueFunction 코어 버그 수정

> 2026-06-19. [[009-p0-glue-decouple-and-f1tenth-loader-done]] 후속. plan_new/008-v4의 **P3**
> (로더 정밀화 + normalizer)를 4축 병렬 조사(opus) → 구현 → 적대적 검증(opus)로 완료.
> + 구조 통일(Diffuser repo를 프로젝트 vendor로 이동, 사용자 지시) + 검증 중 발견한 **ValueFunction
> 원본 off-by-one 버그** 수정. append-only. **★ Diffuser 코드 위치 변경: `~/planning_with_diffusion`
> → `f1tenth_planning_with_diffusion/vendor/diffuser`.**

---

## 0. 한 줄 상태
**P3 완료.** action tier별 v_max raw 역정규화 + lidar min-pool 128 + SafeLimitsNormalizer +
horizon 128/dim_mults (1,4,8) 확정·검증. ValueFunction 원본 버그 수정(value 경로 crash 해소).
Diffuser를 vendor/diffuser로 이동해 Dreamer 패턴 통일(push 일원화). 다음 = P4 학습.

## 1. ★ 구조 변경 — Diffuser를 vendor로 이동 (사용자 지시)
- **이전(이전 세션 설정)**: Diffuser = `~/planning_with_diffusion`(jannerm clone, remote=jannerm).
  글루를 dlacksdn repo로 push 불가 + Dreamer(vendor 내장)와 비일관. 사용자 지적으로 정정.
- **이동**: `~/planning_with_diffusion` → `f1tenth_planning_with_diffusion/vendor/diffuser`.
  jannerm `.git` 제거 → **dlacksdn repo가 원본+글루를 직접 추적**(Dreamer의 `RL_project/vendor/dreamerv3-torch`와 동일).
- **.venv editable 재설정**: `.venv/.../easy-install.pth` + `diffuser.egg-link`의 경로를
  `~/planning_with_diffusion` → `vendor/diffuser`로 sed 교체. import 검증 통과(diffuser.__file__ 새 경로).
- **데이터 경로 불변**: `f1tenth.py`의 `DEFAULT_DATA_DIR`은 RL_project(데이터는 거기). 코드만 이동.
- vendor/diffuser/.gitignore(원본)가 logs(31M)/egg-info/pycache 제외 → repo add 시 78파일(소스+글루).
- ★ **이후 모든 Diffuser 경로는 `vendor/diffuser/` 기준**. 게이트 실행 cwd=`vendor/diffuser`(config.f1tenth import).

## 2. ★ P3 4축 조사 결과 (opus 워크플로, 1차 소스+실측)
### A. action 역정규화 (확정 — 적대적 검증 no_issue)
- npz action = **NormalizeActions의 *입력*(정규화 [-1,1])** — collect가 agent._policy 출력을 env.step
  *전*에 캡처·저장(collect_crash_data.py:148-170 = tools.simulate 미러). 역정규화 1회 정당(이중적용 아님).
- 공식: `steer(a0)=(a0+1)/2*(S_MAX-S_MIN)+S_MIN` (S=±0.4189 대칭, tier 무관) /
  `speed(a1)=(a1+1)/2*(v_max-V_MIN)+V_MIN` (V_MIN=-5.0 고정, v_max=npz per-step). round-trip bit-exact.
- 실측: cap5→5.00 / cap10→10.00 / cap15→15.00 / cap20→20.00 m/s, steer 전 tier 공통 ±0.4189.

### B. normalizer = SafeLimitsNormalizer (Gaussian→교체)
- Gaussian 정규화 OBS 범위 -12.2~33.8이 모델 `clip_denoised=True`의 [-1,1] clamp(diffusion.py:150)
  벗어나 손실 → Limits 계열 필수. SafeLimits는 상수 차원(빔 포화) eps=1 방어가 공짜. 라운드트립 항등.

### C. lidar downsample = 128 + min-pool (64 균등→교체)
- Dreamer는 1080빔 전부 ConvEncoder1D로 학습("각도 구조 보존" 명시) → 64는 손실. 64 균등은 섹터
  14.2%가 최근접 장애물 0.2m+ 과대평가(3m서 빔간격 22.4cm가 차폭 31cm 근접). **128 + 섹터 min-pool**
  (최근접=가장 가까운 벽 보존, 레이싱 안전) → 과대평가 7.9%, 메모리 2.45GB. obs=128+5=**133D**.

### D. hyperparams: horizon 32→128, dim_mults (1,2,4,8)→(1,4,8)
- npz 1 step=정책결정(action_repeat 적용됨, 50step/s). horizon 32=0.64s(너무 짧음). f1tenth는 장기
  경로계획(maze2d 계열) → **horizon 128(2.56s)**. ep 다 ≥128(min143)이라 패딩 의존 없음.
- dim_mults (1,4,8): horizon 키우면 (1,2,4,8)은 채널 급증 + ValueFn horizon 붕괴(→2) → (1,4,8)로 (ValueFn→4).
- 유지: predict_epsilon=False, clip_denoised=True(쌍), n_diffusion_steps=20, action_weight=10, loss_discount=1.

## 3. ★ ValueFunction 원본 off-by-one 버그 발견·수정 (적대적 검증 real_issue → fix)
- **증상**: value 경로 1-step에서 `RuntimeError: mat1 [Bx288] @ mat2 [544x256]`. diffusion은 통과.
- **원인**(temporal.py): blocks 루프가 **모든 블록(is_last 포함)에 Downsample1d(L183)** 적용하나,
  horizon 추적 변수는 `if not is_last`로 가드(L186-187)되어 마지막 블록 //2를 누락 → fc_dim이 2배 과다
  (512 기대 vs 실제 flatten 256). **standard config(horizon 1로 붕괴+max(,1))에선 우연히 가려졌고,
  f1tenth가 (1,4,8)로 horizon 붕괴를 막아 노출**(원본 잠복 버그).
- **수정(fork-patch, 코어지만 버그라 불가피)**: L186-187 가드 제거 + 모든 down 추적을 `(horizon+1)//2`
  (Downsample1d=Conv1d(k3,s2,p1)→ceil(L/2)). mid_down1/2도 동일. **무회귀 검증**: f1tenth(128/(1,4,8))→
  out(2,1) / standard(32/(1,2,4,8))→OK / 홀수 H143→OK. (단독 빌드+forward, torch 실측.)

## 4. ★ value 학습 건전성 (review B major — reward 폭주 방지)
- **reward 미정규화 + ValueFunction normed=False**: 랩완주 +100 보너스 + 장기 progress 누적 →
  discounted return-to-go 수백~수천 → value_l2 회귀 폭주 위험. → **config values `normed=True`**
  (ValueDataset._get_bounds로 value 타깃 [-1,1] 정규화). 적용 완료.

## 5. 남은 review 이슈 (후속 처리)
- **[major] normalizer eval 재fit 불일치**: eval 시 load_diffusion이 SequenceDataset 재구성→SafeLimits를
  현재 디스크 데이터로 재fit. 데이터/downsample 변경 시 train/eval normalizer 어긋남. → **P5 평가 글루에서
  normalizer 통계 pickle 저장+로드 고정** + 데이터셋/`F1TENTH_LIDAR_DOWNSAMPLE` 동결. (지금은 학습이라 미발현.)
- **[minor] use_padding 14% zero-pad 침범**: 학습 윈도 14%가 ep 끝 raw-0 패딩 포함(Limits서 lidar→-1 극단).
  → P4 조정 옵션: use_padding=False(ep<128 없어 손실 0) 또는 max_path_length 축소. 보류(원본 diffuser 성질).

## 6. 검증 요약 (전부 PASS)
- 역정규화 단위(cap5→5..cap20→20, steer ±0.4189) / min-pool (143,1080)→(143,128) / SafeLimits 라운드트립
  allclose / 적재 obs 133·transition 135 / diffusion 1-step loss 0.8807 / ValueFunction forward (2,1) 무회귀 /
  vendor 이동 후 import 동작.

## 7. 다음 (P4)
- **P4 학습**(GPU, vendor/diffuser, run_in_background): diffusion(`train.py --config config.f1tenth`) +
  value(`train_values.py`). 1차 짧은 게이트(예 50k step) loss 곡선 → 폭주/수렴 확인 후 풀 학습.
  value loss corr 모니터(helpers.py ValueLoss). normed=True _get_bounds 비용(794556 인덱스) 첫 적재서 발생.
- 생성 궤적 품질 점검(expert 없이 충돌-위주 데이터로 완주 stitching 되는지) → P5 plan_f1tenth + normalizer 고정.

## 8. 제약/규약
코어 무변경 원칙(temporal/diffusion/helpers/trainer) — 단 ValueFunction은 **원본 버그**라 fork-patch 예외(무회귀).
글루만 수정. ★ push는 사용자 지시 시만. commit·문서 분기마다 자율. 데이터=RL_project(gitignore). 모든 lap=2랩.

# 012 — P5 plan_f1tenth.py 구현 + 평가 venv 통합(중대 발견) + 학습 트림(B)

> 2026-06-20. [[011-p4-launch-dual-critic-and-success-redef]] 후속. P5 평가 코드를 작성하고,
> 작성 전 **평가 venv 통합 리스크를 CPU로 미리 검증**해 중대한 발견(평가는 diffuser venv가 아니라
> RL_project venv에서 돈다)을 확정. + 학습을 (B)로 트림. append-only. 설계=[[007-p5-eval-infra-design]].

---

## 0. 한 줄
plan_f1tenth.py 작성 + `--dry` CPU 스모크 통과. **★평가 실행 venv = RL_project .venv**(diffuser venv 아님 —
gymnasium/ruamel 부재로 env 빌드 불가). RL_project venv에 **tap·GitPython 설치**(torch 2.4.1+cu124 동일 →
체크포인트 호환). 학습은 (B) 트림: diffusion 80k 체크포인트 + value 100k.

## 1. ★ 평가 venv 통합 — 중대 발견 (CPU로 미리 검증, 자율 평가 import-크래시 예방)
007/메모리의 "diffuser venv가 eval 자급(f110_gym 공유)" 가정을 **실측 반증**:
- **diffuser .venv**: diffuser·gym·f110_gym·numba ✅ 이지만 **gymnasium·ruamel 부재** → `build_config`/
  env wrapper 스택(dreamer_f1tenth) import 불가. **평가용으로 부적합.**
- **RL_project .venv**: env 풀스택(gymnasium·ruamel·f110_gym·dreamer env) ✅(P5 plumbing task1서 검증) +
  torch **2.4.1+cu124(diffuser venv와 동일)**·einops·scipy·tqdm ✅. **부족분 = `tap`·`git`(GitPython) 둘뿐**
  → `pip install typed-argument-parser GitPython`(RL_project venv). 이후 diffuser(load_diffusion/
  GuidedPolicy/ValueGuide/config.f1tenth) + env 동시 import 성공(`FULL_INTEGRATION_OK`).
- **결론: 평가는 RL_project .venv + `sys.path`에 vendor/diffuser 추가로 실행**(env 검증된 곳 + diffuser
  순수 torch라 얹힘). torch 버전 동일이라 diffuser venv서 학습한 체크포인트 그대로 로드.

## 2. plan_f1tenth.py (vendor/diffuser/scripts/)
- 로드: `load_diffusion('logs','f1tenth','diffusion/f1tenth_H128_T20')` + `..._values..._d0.99`(D5 정합) →
  `check_compatibility` → `ValueGuide(value.ema)` → `GuidedPolicy(scale=0.1, n_guide_steps=2, t_stopgrad=2,
  sample_fn=n_step_guided_p_sample)`. (plan_guided.py 패턴.) device 기본 cuda:0(평가 시 GPU 비어 있음).
- env: `build_config('f1tenth_Oschersleben')`(**v_max=20**, assert로 강제) + `make_env`. eval_gate 재사용.
- 루프(K=1 MPC): obs → `_downsample_lidar`(로더 함수 재사용=train/eval 일치) min-pool 128 + state5 = 133D →
  `conditions={0: vec}`(1D) → GuidedPolicy → **raw [steer,speed]** → `raw_to_norm(v_max=20)` →
  `env.step({'action': norm})`. per-lap `obs['log_lap_time_s']`, 완주=`info['cause']=='lap_complete'`.
- 2랩 시간 = 완주 ep의 `sum(lap_times)` vs **baseline 107.16s**(beats_baseline 플래그). JSON 저장.
- `--dry`: 모델 미로드 + 더미정책(직진 8m/s)으로 env/변환 CPU 스모크 → **PLAN_F1TENTH_DRY_OK 통과**.
- batch_size=1 기본(GuidedPolicy는 action[0,0]만 써서 결과 동일·속도↑). scale=0.1.

## 3. 검증 상태
- ✅ CPU(모델 없이): import 통합, env 빌드(v_max=20), obs→133D, raw↔norm 왕복 bit-exact(task1),
  `env.step({'action':norm})` 적용(8m/s→state 0.4), `--dry` 1ep 12step 무오류.
- ⏳ 미검증(GPU): `load_diffusion`(dataset 재구성+normalizer 재fit+ckpt 로드) + GuidedPolicy 가이드
  샘플링 풀루프 — **value 학습 종료 후 GPU 비면 실행**(첫 실run이라 잔여 리스크 → episodes=1 스모크 먼저).

## 4. 학습 트림 (사용자 B 선택)
- diffusion: 저장이 40k 간격(state_0, state_40000 디스크) → step 80000 저장까지(~50min) 둔 뒤 정지
  (40k→80k 작업 보존, 더 좋은 체크포인트). **watcher가 state_80000 감지 → 체인 TaskStop → value 100k 발사.**
- value: **100k**(원 200k에서 트림, loss 곡선 saturate). ~9.7h. config values discount=0.99.
- 자율 체인: watcher → TaskStop(체인 bpen1zxgd; ★`;`로 value 200k 자동시작 방지 위해 체인 전체 정지) →
  value 100k → 완료 → plan_f1tenth 평가(episodes 스모크→풀). 총 학습 잔여 ~10.5h.

## 5. 데이터 분포 시각화 (참고)
`_thinking/understand/crash_data_distribution-2.png`(6패널). 다양성·커버리지·0~20m/s 속도 스펙트럼·
고속 단일랩(18–20s) 확인(011 §4). (`-1`은 한글 폰트 깨짐, 무시.)

## 6. 제약/규약
★ git: 이번에 사용자 지시로 commit+push 1회 수행(이후 다시 지시 시만). 코어 무변경(ValueFunction 예외).
RL_project venv에 tap/GitPython 추가(venv는 gitignore라 commit 무관). 모든 lap=2랩. cwd=vendor/diffuser.
normalizer v1=frozen 데이터 재fit(F1TENTH_DATA_DIR·LIDAR_DOWNSAMPLE=128 동결, 추가수집 금지).

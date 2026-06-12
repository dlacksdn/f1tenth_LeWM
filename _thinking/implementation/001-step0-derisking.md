# 001 — Phase 1 Step 0/0b 디리스킹 완료 (환경·스키마·smoke·렌더 스파이크)

> 2026-06-13. 계획 v2(plan/002)의 Phase 1 Step 0(SPOF 해소)와 Step 0b(렌더 스파이크) 실행 기록.
> critic이 꼽은 리스크 Top 3 중 2개(렌더 부재, swm 블랙박스)가 이 분기에서 해소됨.
> append-only.

---

## 0. 요약

- **py3.10 venv + stable-worldmodel 설치 완료** (`f1tenth_LeWM/.venv`, uv 0.11.21 / CPython 3.10.20)
- **공식 tworoom 데이터셋(12GB h5)으로 train.py 레퍼런스 smoke run 통과** — 10 batch 학습 +
  검증 + ckpt 저장까지 전 파이프라인 작동. RTX 4060 Ti 8GB에서 batch 16, ~10.5 it/s
- **실물 h5 스키마 역추출 완료** — 데이터셋 생성 스펙의 SSOT 확보 (아래 §3)
- **렌더 스파이크 성공** — pyglet 없이 map+pose→224×224 ego-centric 래스터 합성 검증(SPIKE_OK)
- 논문(raws/LeWorldModel.pdf) 확인으로 frameskip 의미·권장 데이터 규모 미검증 항목 해소
- **차체 물리값 무변경** (사용자 절대 제약): 이 분기는 f1tenth env 코드를 일절 건드리지 않음

## 1. 환경 세팅 (Step 0-1)

- uv 설치(`~/.local/bin/uv`) → `uv venv --python=3.10 f1tenth_LeWM/.venv` (managed CPython 3.10.20)
- `uv pip install "stable-worldmodel[train,env]"` → 396 패키지 (torch cu12 포함)
- **함정 발견**: swm `data/formats/hdf5.py:11`이 `import hdf5plugin` 하는데 의존성 미선언 →
  수동 설치 필요 (`uv pip install hdf5plugin`, 6.0.0). 없으면 pixels(blosc) 읽기 실패:
  "can't open directory /usr/local/lib/plugin"
- 시스템에 zstd 바이너리 없음 → `uv run --with zstandard` 스트리밍으로 tar.zst 해제
- GPU: RTX 4060 Ti 8GB (논문 L40S 48GB 대비 작음 → 본 학습 시 batch_size 조정 필요)

## 2. 데이터셋 경로 규약 (README와 다름 — 함정)

- README "Place .h5 under `$STABLEWM_HOME`"은 **부정확**. 실제 코드(`swm/data/utils.py:get_cache_dir`,
  `load_dataset`)는 `$STABLEWM_HOME/datasets/<name>` 을 본다 → `stablewm_home/datasets/tworoom.h5`
- env var는 `STABLEWM_HOME`(기본 `~/.stable_worldmodel`). 본 프로젝트는
  `STABLEWM_HOME=/home/dlacksdn/f1tenth_LeWM/stablewm_home` 사용 (~ 폴더 미생성 원칙)
- ckpt는 `$STABLEWM_HOME/checkpoints/<run_name>/weights_epoch_N.pt` (SaveCkptCallback →
  swm `save_pretrained`). lightning 자체 ckpt는 `~/.cache/stable-pretraining/runs/<날짜>/...`

## 3. 실물 h5 스키마 (tworoom 공식, 데이터셋 생성 스펙 SSOT)

**flat row-major** 구조: 전 에피소드의 transition을 이어붙인 1차원 배열 + 인덱스 테이블.

| key | shape | dtype | 비고 |
|---|---|---|---|
| pixels | (N, 224, 224, 3) | uint8 | **HWC**, chunks (100,224,224,3), **blosc 압축**(filter 32001) |
| action | (N, 2) | float32 | per-step raw action |
| proprio | (N, 2) | float32 | keys_to_load에 포함, encode 미사용 |
| observation | (N, 10) | float64 | env 고유 (우리는 불필요) |
| ep_idx | (N,) | int32 | row→episode 매핑 |
| step_idx | (N,) | int64 | episode 내 step 번호 |
| ep_len | (E,) | int32 | E=에피소드 수(10000) |
| ep_offset | (E,) | int64 | episode 시작 row |
| reward/terminated/truncated 등 | (N,) | - | env 고유 잉여 컬럼 (로더는 keys_to_load만 사용) |

N=920,809 (10k ep × 평균 92 step). 그 외 컬럼(pos_agent 등)은 tworoom 고유 — **필수 아님**.

### 로더 윈도잉 (load_dataset(num_steps=4, frameskip=5) 실측)

- len(dataset) = 730,809 윈도우 (= N − E×(num_steps−1)×frameskip 근사)
- 샘플 dict: `pixels (4,3,224,224) uint8` (**CHW로 변환됨**), `action (4,10) float32`
  (= 4 frame × **frameskip 5 × action_dim 2 평탄화**), `proprio (4,2)`
- 즉 h5엔 매 step 저장, 로더가 5-step 간격 프레임 4장 + 사이 action 5개씩 묶음
- f1tenth 대응: action (T,2) → 블록당 (10,) ✓. proprio ← state(5) 그대로 가능 ✓

## 4. 레퍼런스 smoke run (Step 0-3)

```bash
STABLEWM_HOME=.../stablewm_home .venv/bin/python train.py data=tworoom \
  loader.batch_size=16 num_workers=2 trainer.max_epochs=1 \
  +trainer.limit_train_batches=10 +trainer.limit_val_batches=2
```
- validate/loss 0.656 (pred_loss 0.049 + 0.09×sigreg 6.73) — loss 산식 코드와 일치 확인
- ckpt 저장: `stablewm_home/checkpoints/lewm/weights_epoch_1.pt` (206.8MiB lightning ckpt 별도)
- `${eval:}` resolver 정상 (m-6 해소), wandb는 launcher/local.yaml에서 기본 disabled

## 5. 렌더 스파이크 (Step 0b) — `spikes/render_spike.py`

- **pose는 npz에 없음** (critic 가정 어긋남): state(5)=[vel_x, vel_y, ang_z, prev_steer, prev_speed].
  스파이크는 centerline csv(s,x,y,tx,ty)에서 pose 합성 (θ=atan2(ty,tx))
- ⇒ **본 수집 시 raw env에서 (poses_x, poses_y, poses_theta) 병행 기록 필수** (Phase 3-1 스펙에 반영)
- 합성: map_easy3.png(1502×1646, 0.02m/px, origin [-2.7,-19.32]) → pose 중심 crop → heading-up
  회전 → 22.4m 시야 → 224×224 RGB + 차량 삼각형 마커. numpy/PIL만, headless-safe, 결정적
- 결과: 6개 pose 전부 정상 생성 (`spikes/out/ego_s*.png`), SPIKE_OK
- 남은 설계 항목(Phase 2 결정①에서): 시야 폭(22.4m 적정성), 벽 선 두께/주행영역 채우기 등
  시인성, fill 색상 의미론

## 6. 논문 확인 (LeWorldModel.pdf, 미검증 항목 해소)

- frameskip=5: "grouping consecutive actions between frames into a single action block",
  배치 = 4 frame + 4×(5 action) 블록, 224×224
- 데이터 규모: TwoRoom 10k ep×92, PushT 20k×196, Cube/Reacher 10k×200 — **paper 스케일 ≈ 1M~4M
  transition**. v1 추정(300ep)보다 훨씬 큼 → 결정②에서 규모 재산정 필요
- 학습량: **10 epoch이면 충분** (config 기본 100과 다름), 단일 GPU
- 수집 정책: "pseudo-expert or exploratory, as long as they sufficiently cover the environment
  dynamics" → diversity snapshot 믹스 전략 부합
- ⚠️ **신규 리스크**: 한계 절 — "limited data diversity can affect the effectiveness of the SIGReg
  regularization" (TwoRoom이 최악 사례). f1tenth ego-view는 시각 다양성 제한적 →
  다양한 속도/주행선 policy 믹스가 필수 요건으로 승격. Phase 2 결정②에 반영
- MPC: horizon 5 × frameskip 5 = 25 env step lookahead, receding=5(전부 실행 후 재계획),
  CEM 30 iter(PushT)/10 iter(기타), top-30, var 1.0

## 7. 다음 분기

- Phase 1 잔여: swm 내부 코드 분석 (HDF5Dataset 윈도잉 경계 처리, WorldModelPolicy/CEMSolver
  정규화·역정규화 경로, swm.World API) — 설치돼 있으므로 소스 직접 분석 가능
- Phase 2 설계 결정 4개 (모달리티는 스파이크 성공으로 (a) 우세, 공식 결정은 Phase 2에서)
- git: 초기 커밋 ef8000b, remote `git@github.com:dlacksdn/f1tenth_LeWM.git` 설정됨.
  **GitHub repo 생성은 사용자 작업 대기 중** — 생성 즉시 push

# 004 — f1tenth_LeWM 환경 세팅 현황 (002 대체)

> 2026-06-13. 001~003은 dreamer 메인 프로젝트에서 복사한 재활용 문서로, 특히 002의 폴더
> 구조/venv 서술은 옛 프로젝트 기준이라 본 프로젝트와 맞지 않음. append-only 원칙에 따라
> 002를 수정하지 않고 본 문서가 **현재 유효한 세팅 기록**으로 대체한다.

---

## 폴더 구조 (이 프로젝트 기준)

```
~/
├── f1tenth_LeWM/                      ← 본 프로젝트 루트 (git: github.com/dlacksdn/f1tenth_LeWM)
│   ├── .venv/                         ← Python 3.10.20 (uv managed), LeWM 학습/평가용
│   ├── stablewm_home/                 ← $STABLEWM_HOME (gitignore)
│   │   ├── datasets/tworoom.h5        ← 공식 레퍼런스 데이터셋 (12GB)
│   │   └── checkpoints/lewm/          ← 학습 ckpt 저장 위치
│   ├── spikes/render_spike.py         ← Step 0b 렌더 스파이크
│   └── _thinking/{plan,analysis,implementation,env_setting,raws}/
├── le-wm/                             ← 공식 LeWM 코드 (무수정 clone)
└── f1tenth_RL_project/                ← Dreamer 프로젝트 (offline 데이터셋 생성원)
    ├── .venv/                         ← Python 3.8.10 (시뮬레이터/rollout용)
    ├── runs/stage1_map_easy3/         ← policy snapshot (diversity bin 9개 + best 6.1s)
    ├── runs/stage2_oschersleben/      ← (best 16.6s)
    └── maps/                          ← centerline csv (easy3, Oschersleben)
```

## 두 venv 역할 분담 (핵심 원칙)

| venv | Python | 용도 | 비고 |
|---|---|---|---|
| `f1tenth_LeWM/.venv` | 3.10.20 | LeWM 학습·planning·h5 읽기/쓰기 | `uv venv --python=3.10` |
| `f1tenth_RL_project/.venv` | 3.8.10 | f110 시뮬레이터 rollout·데이터 수집 | 기존 그대로, **무변경** |

- **`.h5` 파일이 두 환경의 유일한 경계 인터페이스** (수집 py3.8 → 학습 py3.10)
- f1tenth **차체 물리값 절대 불변** (사용자 제약) — rollout 시 env 코드/파라미터 무변경

## 설치 내역 (py3.10 venv)

```bash
# uv 설치 (1회)
curl -LsSf https://astral.sh/uv/install.sh | sh   # → ~/.local/bin/uv

cd ~/f1tenth_LeWM
uv venv --python=3.10 .venv
uv pip install -p .venv "stable-worldmodel[train,env]"   # 396 패키지, torch cu12 포함
uv pip install -p .venv hdf5plugin   # ★ 필수 — swm 의존성 미선언 버그 (blosc h5 읽기)
```

## 환경 변수 / 실행 규약

```bash
export STABLEWM_HOME=/home/dlacksdn/f1tenth_LeWM/stablewm_home
# 데이터셋은 $STABLEWM_HOME/datasets/<name>.h5 에 둬야 함 (README의 "$STABLEWM_HOME 바로 아래"는 부정확)
cd ~/le-wm
~/f1tenth_LeWM/.venv/bin/python train.py data=tworoom   # 예시
```

## 하드웨어

- GPU: RTX 4060 Ti 8GB (WSL2, Ubuntu 20.04) — 논문(단일 L40S)보다 작음.
  smoke 실측: batch 16에서 ~10.5 it/s. 본 학습 전 batch 스윕 1회 예정
- 디스크: ~925GB 여유 (tworoom 12GB + 향후 f1tenth 데이터셋 수십 GB 감안 충분)

## 알려진 함정 (재현 시 주의)

1. `hdf5plugin` 수동 설치 필수 (위 참조) — 없으면 pixels(blosc) 읽기 실패
2. 데이터셋 경로는 `datasets/` 하위 (README와 다름)
3. 시스템에 zstd 없음 — tar.zst 해제는 `uv run --with zstandard` 스트리밍으로 처리
4. lightning 부산물이 `~/.cache/stable-pretraining/runs/`에 생성됨 (프로젝트 밖, 정리 대상 인지)

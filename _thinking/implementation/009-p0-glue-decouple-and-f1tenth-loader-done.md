# 009 — P0 글루 디커플 + f1tenth 로더 + 1-step 게이트 완료

> 2026-06-19. [[008-gpu-batch-measured-no-gain-and-p2-done]] 후속. plan_new/008-v4의 **P0**
> (글루 디커플 + 더미/실 로더 train.py 1-step)를 실행 완료. 게이트 A/B/C/D 전부 PASS.
> 지도 = [[006-glue-correction-and-data-contract]](analysis) §2 + [[005-diffuser-glue-touchpoints]].
> append-only. **Diffuser repo = `~/planning_with_diffusion`(jannerm clone), 데이터 = RL_project.**

---

## 0. 한 줄 상태
**P0 완료.** Diffuser 코어 import 통과 + f1tenth 로더로 767 ep 적재(69D) + 1-step
forward/backward loss(0.7201) 산출 + NullRenderer pickle. **코어(temporal/diffusion/helpers/
trainer) 무변경**, 글루만 수정. P0가 P3 로더 초안까지 겸함(정밀화는 P3 잔여). 다음 = P3/P4.

## 1. import 디커플 (게이트 A — TemporalUnet import)
체인: `temporal → helpers:10(import diffuser.utils) → utils/__init__:6(from .rendering import *)`.
**4파일 top-level heavy import를 try/except guard**(코어 무변경, 글루만):
- `utils/rendering.py`: imageio·mujoco_py·`.video`·d4rl guard (클래스 메서드 내부서만 실사용 → top guard로 충분).
- `utils/video.py`: `import skvideo.io` guard (colab.py→video.py 체인이 코어까지 죽임 — 실측 발견).
- `datasets/d4rl.py:23`: `import d4rl` try/except (전수 차단, f1tenth 로더라 d4rl API 미호출).
- `datasets/buffer.py:12`: `np.int → np.int64` (numpy 1.24.4 제거).
→ `from diffuser.models.temporal import TemporalUnet` 등 PASS(colab 경고는 무해 자체 except).

## 2. f1tenth 로더 (게이트 B — 적재)
- **신설 `datasets/f1tenth.py`**: `load_f1tenth_environment`(더미 env: .seed/._max_episode_steps/
  .name + 데이터경로/downsample) + `f1tenth_sequence_dataset`(npz→에피소드 dict yield) +
  **`F1tenthSequenceDataset`/`F1tenthValueDataset`**(SequenceDataset/ValueDataset 서브클래스).
- **`sequence.py` hook**(backward-compat): `_load_environment`/`_sequence_dataset`를 staticmethod로
  노출(기본=원래 d4rl 함수 → locomotion 등 100% 불변), 서브클래스가 f1tenth로 교체. 코어 무변경.
- **`datasets/__init__.py`/`utils/__init__.py`** 노출 추가.
- **표현 v1 = concat(downsample(lidar, 64), state) = 69D**. ★ full lidar(1085D)는 ReplayBuffer
  사전할당 `(max_n_episodes×max_path_length×dim)`이 ~19GB(실측 ep max 5829)라 **불가** →
  균등각 다운샘플 64 → 1.23GB. (다운샘플 수/데이터경로 = 환경변수 override 가능.)
- **계약 키**: observations / actions(npz 정규화 그대로, P0) / rewards / terminals(=is_terminal) /
  **timeouts(=is_last & ~is_terminal)**. 충돌=terminal·timeouts전부False, 완주=non-terminal·timeout마지막.
- 게이트 B 실측: 767 ep 적재(22.5s), obs_dim=69, act_dim=2, len(ds)=795422, traj (32,71).

## 3. 1-step + 렌더러 (게이트 C/D)
- **`config/f1tenth.py` 신설**: base{diffusion,values,plan}. loader=F1tenth*, renderer=NullRenderer,
  normalizer=GaussianNormalizer(P0 잠정; P3 SafeLimits 검토), clip_denoised=True, sample_freq=0,
  max_path_length=5839, termination_penalty=None(values). horizon=32, n_diffusion_steps=20.
- **신설 `utils/null_renderer.py`**: `NullRenderer`(render/composite/__call__ no-op) — MuJoCoRenderer 대체.
- 게이트 C(`/tmp/p0_gate.py`, train.py L108-112 격리): **GATE_C_PASS loss=0.7201**, transition_dim=71,
  TemporalUnet 채널 [(71,32),(32,64),(64,128),(128,256)].
- 게이트 D: NullRenderer Config 인스턴스화+pickle OK, sampling.policies/guides import OK.

## 4. ★ 함정/교훈 (1차 소스 — 인수인계)
- **`pkill -f "train.py"`/`"p0_gate.py"`가 자기 bash 명령 라인을 매칭→self-kill(exit 144)**. 디버깅을
  한참 헤맴. 프로세스 정리는 bracket-trick(`[t]rain.py`) 또는 PID 직접. **pkill에 자기 문자열 금지.**
- **샌드박스 foreground + CUDA = exit 144**(추정)였으나 실제 주범은 위 self-kill. GPU 1-step은
  `run_in_background`(pkill 없이)로 정상 exit 0. (RL collect도 background로 GPU 성공한 패턴과 동일.)
- **batchify(`utils/arrays.py:35`)는 `to_torch`→모듈상수 DEVICE(cuda)로 하드코딩 이동** → 1-step은
  GPU 불가피(device='cpu' config여도 batch는 cuda). P4 학습도 GPU.
- **ReplayBuffer 사전할당이 max_n_episodes 전체**(buffer.py:60) → F1tenthSequenceDataset가
  max_n_episodes를 실제 ep 수(+10)로 자동 설정(기본 10000이면 16GB OOM).
- **timeouts assert(buffer.py:79)**는 termination_penalty 시 충돌 ep(timeouts전부False)만 검사 → 통과.

## 5. 다음 (P3 정밀화 + P4 학습)
- **P3 잔여**(로더 골격은 P0에서 완성): ① action **tier별 v_max raw 역정규화**(현재 정규화 그대로 —
  tier 간 스케일 불일치, +1이 cap5→5/cap20→20 일관화 필요) ② normalizer 선택(SafeLimitsNormalizer
  검토) + **라운드트립 검산** ③ downsample 수 확정(64 잠정) ④ pose는 현재 obs 미포함(centerline 피처 v2).
- **P4**: diffusion + value 학습(GPU) + 생성 궤적 품질 점검. value=F1tenthValueDataset(reward
  discounted return-to-go) + termination_penalty=None.
- **P5/P6**: plan_f1tenth.py(GuidedPolicy↔f110_gym) + cap-5 baseline 대비 2랩 lap time 평가.

## 6. 제약/규약 (불변)
코어(temporal/diffusion/helpers/trainer) 무변경. 글루만 수정. ★ push는 사용자 지시 시만.
commit·문서 분기마다 자율. 데이터=RL_project/runs/crash_data(gitignore). 모든 lap=2랩.

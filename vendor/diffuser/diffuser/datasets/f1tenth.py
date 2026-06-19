"""f1tenth offline 데이터 로더 (Diffuser P0/P3 글루).

RL_project에서 수집한 충돌/완주 npz(``runs/crash_data/cap*``)를 Diffuser
``SequenceDataset``이 소비하는 에피소드 dict로 변환한다. d4rl의
``load_environment``/``sequence_dataset``를 대체(``SequenceDataset``의
``_load_environment``/``_sequence_dataset`` hook 오버라이드). **코어 무변경.**

표현(P3): observations = concat(min-pool-downsample(lidar, 128), state) = 133D.
  full lidar(1080)는 ReplayBuffer 사전할당 ~19GB라 불가 → 섹터 min-pool 다운샘플(기본 128,
  최근접 장애물 보존) → 133D(~2.5GB). (다운샘플 수/데이터 경로 환경변수 override.)

계약 키(d4rl.sequence_dataset 미러, buffer.add_path 요구):
  - observations : concat(min-pool-downsample(lidar,128), state)
  - actions      : **P3 역정규화 적용** — npz action(tier-상대 [-1,1])을 tier별 v_max로
                   raw 물리값(steer rad, speed m/s) 복원(+1이 cap5→5/cap20→20m/s 일관화).
  - rewards      : npz reward.
  - terminals    : is_terminal (충돌=True, 완주=False=non-terminal → value 부트스트랩 가능).
  - timeouts     : is_last & ~is_terminal (완주=마지막 True, 충돌=전부 False).
                   buffer.py:79 `assert not timeouts.any()`는 termination_penalty 시
                   충돌 ep(timeouts 전부 False)만 검사 → 통과.
"""
import os
import glob
import numpy as np

from .sequence import SequenceDataset, ValueDataset

DEFAULT_DATA_DIR = os.environ.get(
    'F1TENTH_DATA_DIR', '/home/dlacksdn/f1tenth_RL_project/runs/crash_data')
DEFAULT_DOWNSAMPLE = int(os.environ.get('F1TENTH_LIDAR_DOWNSAMPLE', '128'))

# action 물리 상수 (f1tenth_env.py:35-36 SSOT, 조사 A 확정). NormalizeActions 역식용.
# action[0]=steer(rad), action[1]=speed(m/s). steer/V_MIN은 tier 무관 고정, v_max만 tier별.
S_MIN, S_MAX = -0.4189, 0.4189   # steering rad
V_MIN = -5.0                      # speed 하한 m/s (reverse), 캡 정책도 고정


class F1tenthEnv:
    """SequenceDataset이 기대하는 최소 env 인터페이스(.seed/._max_episode_steps/.name)
    + 데이터 경로/다운샘플 보관. d4rl OfflineEnv 대체용 더미."""

    def __init__(self, data_dir, downsample, name='f1tenth', max_episode_steps=10 ** 7):
        self.data_dir = data_dir
        self.downsample = downsample
        self.name = name
        self._max_episode_steps = max_episode_steps

    def seed(self, seed=None):
        return None


def _resolve_data_dir(name):
    """env 인자가 실제 디렉토리면 그대로, 아니면 DEFAULT_DATA_DIR(라벨로 취급)."""
    if isinstance(name, str) and os.path.isdir(name):
        return name
    return DEFAULT_DATA_DIR


def _npz_files(data_dir):
    """실데이터 tier(cap5/cap5_full/cap10/cap10_full/cap15/cap20)만. _probe/_smoke/빈 폴더 제외."""
    return sorted(glob.glob(os.path.join(data_dir, 'cap*', '*.npz')))


def load_f1tenth_environment(name='f1tenth', downsample=DEFAULT_DOWNSAMPLE):
    return F1tenthEnv(_resolve_data_dir(name), downsample,
                      name=name if isinstance(name, str) else 'f1tenth')


def _downsample_lidar(lidar, n):
    """lidar (T,1080) → (T,n). 섹터별 min-pool(최근접 장애물 보존 = 레이싱 안전, 조사 C).

    균등 인덱스는 빔 사이 최근접점을 누락(64빔서 섹터 14.2%가 0.2m+ 과대평가, 3m서 빔간격
    22.4cm가 차폭 31cm 근접). 각 섹터의 최소(=가장 가까운 벽/장애물)를 취해 안전쪽 보수화.
    정규화 [0,1]서 0=충돌·1=30m clear라 min이 최근접과 정합. Dreamer는 1080빔 전부 학습
    (ConvEncoder1D)했으나 메모리(full 1080D=19GB)로 다운샘플 불가피 → 128(2.5GB)로 손실 최소화.
    """
    if n is None or n >= lidar.shape[-1]:
        return lidar
    B = lidar.shape[-1]
    edges = np.linspace(0, B, n + 1).round().astype(int)
    out = np.empty((lidar.shape[0], n), dtype=lidar.dtype)
    for j in range(n):
        out[:, j] = lidar[:, edges[j]:edges[j + 1]].min(axis=1)
    return out


def _denormalize_action(action, v_max):
    """tier-상대 정규화 action [-1,1] → raw 물리값 (조사 A 확정, NormalizeActions 역식).

    action (T,2)=[steer_norm, speed_norm]. v_max (T,) per-step(npz 'v_max') 또는 스칼라.
      steer(rad) = (a0+1)/2*(S_MAX-S_MIN)+S_MIN        # tier 무관 고정 (= a0*0.4189)
      speed(m/s) = (a1+1)/2*(v_max-V_MIN)+V_MIN         # V_MIN=-5 고정, v_max만 tier별
    → 모든 tier의 +1 speed가 그 tier 실제 m/s로 통일(cap5→5, cap20→20). steer는 tier 공통.
    """
    a = np.asarray(action, dtype=np.float32)
    vmax = np.asarray(v_max, dtype=np.float32).reshape(-1)        # (T,)
    raw = np.empty_like(a)
    raw[:, 0] = (a[:, 0] + 1.0) * 0.5 * (S_MAX - S_MIN) + S_MIN   # steer (rad)
    raw[:, 1] = (a[:, 1] + 1.0) * 0.5 * (vmax - V_MIN) + V_MIN    # speed (m/s)
    return raw


def f1tenth_sequence_dataset(env, preprocess_fn=None):
    """npz → 에피소드 dict iterator (d4rl.sequence_dataset 계약 미러)."""
    downsample = getattr(env, 'downsample', DEFAULT_DOWNSAMPLE)
    data_dir = getattr(env, 'data_dir', DEFAULT_DATA_DIR)
    for f in _npz_files(data_dir):
        d = np.load(f)
        lidar = _downsample_lidar(d['lidar'].astype(np.float32), downsample)
        state = d['state'].astype(np.float32)
        is_terminal = d['is_terminal'].astype(bool)
        is_last = d['is_last'].astype(bool)
        v_max = d['v_max'] if 'v_max' in d.files else np.full(len(d['action']), 20.0, np.float32)
        episode = {
            'observations': np.concatenate([lidar, state], axis=-1),
            'actions': _denormalize_action(d['action'], v_max),   # P3: raw [steer rad, speed m/s]
            'rewards': d['reward'].astype(np.float32),
            'terminals': is_terminal,
            'timeouts': is_last & ~is_terminal,
        }
        if preprocess_fn is not None:
            episode = preprocess_fn(episode)
        yield episode


def _auto_max_n_episodes(env):
    """ReplayBuffer 사전할당이 (max_n_episodes×max_path_length×dim)이라 실제 ep 수로 한정."""
    return len(_npz_files(_resolve_data_dir(env))) + 10


class F1tenthSequenceDataset(SequenceDataset):
    """d4rl 대신 f1tenth 로더 사용(hook 교체). max_n_episodes는 데이터 ep 수로 자동."""

    _load_environment = staticmethod(load_f1tenth_environment)
    _sequence_dataset = staticmethod(f1tenth_sequence_dataset)

    def __init__(self, env='f1tenth', max_n_episodes=None, **kwargs):
        if max_n_episodes is None:
            max_n_episodes = _auto_max_n_episodes(env)
        super().__init__(env=env, max_n_episodes=max_n_episodes, **kwargs)


class F1tenthValueDataset(ValueDataset):
    """value 학습용. ValueDataset(=SequenceDataset 서브클래스) + f1tenth hook."""

    _load_environment = staticmethod(load_f1tenth_environment)
    _sequence_dataset = staticmethod(f1tenth_sequence_dataset)

    def __init__(self, env='f1tenth', max_n_episodes=None, **kwargs):
        if max_n_episodes is None:
            max_n_episodes = _auto_max_n_episodes(env)
        super().__init__(env=env, max_n_episodes=max_n_episodes, **kwargs)

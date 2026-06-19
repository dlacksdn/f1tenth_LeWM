import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## 실험 폴더 자동 라벨용 (planning)
args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

## f1tenth glue (P3): 표현=concat(min-pool-downsample(lidar,128), state)=133D, max_path_length=실측 max(5829)+여유.
## renderer=NullRenderer + sample_freq=0(헤드리스). loader=F1tenth*(d4rl 대체).
## P3 확정(조사 4축): normalizer=SafeLimitsNormalizer(clip_denoised=True의 [-1,1] clamp 정합 + 상수차원
## eps 안전), horizon=128(50step/s=2.56s 장기 계획, maze2d 계열), dim_mults=(1,4,8)(ValueFn horizon
## 붕괴 방지: 128→4 vs (1,2,4,8)의 →2). action은 로더에서 tier별 v_max raw 역정규화 완료.
base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 128,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'attention': False,
        'renderer': 'utils.NullRenderer',

        ## dataset
        'loader': 'datasets.F1tenthSequenceDataset',
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 5839,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/f1tenth',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 128,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.NullRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': None,
        'normed': True,   # P3 review: reward 미정규화(랩완주 +100) → return-to-go 수백~수천 → value_l2 폭주 방지(타깃 [-1,1])

        ## dataset
        'loader': 'datasets.F1tenthValueDataset',
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 5839,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/f1tenth',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 6000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 128,
        'n_diffusion_steps': 20,

        ## value function
        ## D5 fix(검수): value는 values.discount=0.99로 저장(exp=...d0.99)되므로 value_loadpath의
        ## {discount}가 0.99여야 P5 eval에서 로드됨(0.997이면 d0.997 폴더 못 찾아 로드 실패). plan
        ## 블록은 eval 전용이라 실행 중 학습엔 무영향.
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/f1tenth_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/f1tenth_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },
}

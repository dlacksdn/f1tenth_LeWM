"""NullRenderer — f1tenth glue (P0).

MuJoCoRenderer(mujoco_py 의존) 대체. 헤드리스 학습/평가에서 모든 렌더 호출을 no-op로
삼킨다. config ``renderer='utils.NullRenderer'`` + ``sample_freq=0``으로 사용해
Trainer의 render_reference/render_samples 경로가 절대 실호출되지 않게 한다(이중 안전).
"""


class NullRenderer:
    def __init__(self, env=None, *args, **kwargs):
        self.env = env

    def render(self, *args, **kwargs):
        return None

    def renders(self, *args, **kwargs):
        return None

    def composite(self, savepath, paths, *args, **kwargs):
        return None

    def render_plan(self, *args, **kwargs):
        return None

    def render_rollout(self, *args, **kwargs):
        return None

    def __call__(self, *args, **kwargs):
        return None

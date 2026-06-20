#!/usr/bin/env python3
"""Diffuser 정책으로 차가 굴러가는 걸 실시간 창으로 본다 (f110 render, WSLg).

watch_drive.py의 렌더 패턴(find_f110 → F110Env.render('human')) + GuidedPolicy(P5).
환경/물리 무수정 — render() 호출만 끼움.

실행(★RL_project venv, cwd=vendor/diffuser):
  cd /home/dlacksdn/f1tenth_planning_with_diffusion/vendor/diffuser
  F1TENTH_LIDAR_DOWNSAMPLE=128 /home/dlacksdn/f1tenth_RL_project/.venv/bin/python \
      scripts/watch_f1tenth.py --K 10 --episodes 3
옵션: --K 1(매 step 재계획=감속/스핀 모드) / --K 10(플랜 추종=고속 주행 모드) / --scale 0.1 / --mode human
창을 닫으면 관람 종료.
"""
import argparse
import os
import sys

RLROOT = "/home/dlacksdn/f1tenth_RL_project"
sys.path.insert(0, os.path.join(RLROOT, "scripts"))
sys.path.insert(0, os.path.join(RLROOT, "vendor", "dreamerv3-torch"))
sys.path.insert(0, "/home/dlacksdn/f1tenth_planning_with_diffusion/vendor/diffuser")

import numpy as np  # noqa: E402

S_MIN, S_MAX, V_MIN = -0.4189, 0.4189, -5.0


def r2n(raw, vm=20.0):
    return np.clip(np.array([2 * (raw[0] - S_MIN) / (S_MAX - S_MIN) - 1,
                             2 * (raw[1] - V_MIN) / (vm - V_MIN) - 1], np.float32), -1, 1)


def find_f110(env):
    """make_env 체인을 .env/._env로 따라 내려가 내부 F110Env를 찾는다(watch_drive 패턴)."""
    from f110_gym.envs.f110_env import F110Env
    e, seen = env, set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, F110Env):
            return e
        e = getattr(e, "env", None) or getattr(e, "_env", None)
    raise RuntimeError("체인에서 F110Env를 못 찾음 (render 대상 없음)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--K", type=int, default=10, help="플랜 K step 실행 후 재계획(1=매step=스핀모드, 10=고속주행모드)")
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--mode", default="human", choices=["human", "human_fast"])
    ap.add_argument("--max_steps", type=int, default=6000)
    args = ap.parse_args()

    from diffuser.utils import load_diffusion
    from diffuser.sampling import GuidedPolicy, ValueGuide, n_step_guided_p_sample
    from diffuser.datasets.f1tenth import _downsample_lidar
    from eval_gate import build_config
    from dreamer import make_env

    def o2c(obs):
        ld = _downsample_lidar(obs["lidar"].reshape(1, -1).astype(np.float32), 128)[0]
        return np.concatenate([ld, obs["state"].astype(np.float32)])

    diff = load_diffusion("logs", "f1tenth", "diffusion/f1tenth_H128_T20")
    val = load_diffusion("logs", "f1tenth", "values/f1tenth_H128_T20_d0.99")
    policy = GuidedPolicy(guide=ValueGuide(val.ema), diffusion_model=diff.ema,
                          normalizer=diff.dataset.normalizer, preprocess_fns=[],
                          sample_fn=n_step_guided_p_sample, scale=args.scale,
                          n_guide_steps=2, t_stopgrad=2, scale_grad_by_std=True)
    cfg = build_config("f1tenth_Oschersleben")
    env = make_env(cfg, "eval", 0)
    f110 = find_f110(env)
    print(f"[watch] K={args.K} scale={args.scale} — 실시간 창을 띄웁니다(창을 닫으면 종료). "
          f"diffusion 샘플링이라 느릴 수 있음(K step마다 잠깐 멈춤).", flush=True)
    for ep in range(args.episodes):
        obs = env.reset(); length = 0; cause = None; done = False
        while length < args.max_steps and not done:
            _, traj = policy({0: o2c(obs)}, batch_size=args.batch_size, verbose=False)
            plan = np.asarray(traj.actions[0])
            for j in range(min(args.K, len(plan))):
                a = plan[j]
                obs, r, done, info = env.step({"action": r2n(a)}); length += 1
                try:
                    f110.render(mode=args.mode)
                except Exception as exc:
                    print(f"[watch] 창 닫힘/렌더 종료: {exc}", flush=True)
                    env.close(); return
                if length % 25 == 0:
                    print(f"  ep{ep+1} step{length}: speed={a[1]:5.1f} m/s  steer={a[0]:+.2f}", flush=True)
                if done:
                    cause = info.get("cause"); break
        print(f"[watch] ep{ep+1} 종료: cause={cause} length={length} (2랩완주={cause=='lap_complete'})", flush=True)
    env.close()


if __name__ == "__main__":
    main()

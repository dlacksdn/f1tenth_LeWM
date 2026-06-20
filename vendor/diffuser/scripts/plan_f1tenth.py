#!/usr/bin/env python3
"""P5: Diffuser(GuidedPolicy) ↔ f110_gym 2랩 평가 (plan_f1tenth).

설계 = _thinking/analysis/007-p5-eval-infra-design.md (+ 하단 정정: V_MAX=20) / understand/001.
- diffusion + value 모델을 load_diffusion으로 로드 → ValueGuide → GuidedPolicy(scale, n_guide_steps,
  t_stopgrad, sample_fn=n_step_guided_p_sample). (plan_guided.py 패턴.)
- env = f110_gym Oschersleben (eval_gate.build_config v_max=20 + make_env, 그대로 재사용).
- 매 step(K=1 receding-horizon MPC): obs(lidar1080+state5) → 섹터 min-pool 128(로더 _downsample_lidar
  재사용) → 133D conditions{0:obs} → GuidedPolicy → **raw [steer rad, speed m/s]** → raw_to_norm(v_max=20)
  → env.step({'action': norm}). (NormalizeActions 역, S4 정정 = V_MAX 20 고정.)
- 2랩 완주(info['cause']=='lap_complete') + per-lap obs['log_lap_time_s'] → 2랩 lap time vs baseline 107.16s.

★ 실행 venv = **RL_project .venv**(env+diffuser 통합 검증; torch 2.4.1+cu124 동일; tap·GitPython 설치 필요).
  diffuser .venv 아님(gymnasium/ruamel 부재로 env 빌드 불가 — impl/012 참조).
★ cwd = vendor/diffuser (load_diffusion의 'logs/...' 상대경로 + config import).
★ normalizer v1 = frozen 데이터 재fit(F1TENTH_DATA_DIR + F1TENTH_LIDAR_DOWNSAMPLE=128 동결 전제).

실행:
  cd /home/dlacksdn/f1tenth_planning_with_diffusion/vendor/diffuser
  F1TENTH_LIDAR_DOWNSAMPLE=128 /home/dlacksdn/f1tenth_RL_project/.venv/bin/python \
      scripts/plan_f1tenth.py --episodes 5
  # CPU 스모크(모델 미로드, env/변환만): ... scripts/plan_f1tenth.py --dry
"""
import argparse
import json
import os
import pathlib
import sys

# --- sys.path: env(dreamer) + diffuser(vendor) ---
RLROOT = "/home/dlacksdn/f1tenth_RL_project"
sys.path.insert(0, os.path.join(RLROOT, "scripts"))                       # eval_gate
sys.path.insert(0, os.path.join(RLROOT, "vendor", "dreamerv3-torch"))      # dreamer.make_env, envs
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))   # vendor/diffuser (diffuser, config)

import numpy as np  # noqa: E402

# action 물리 상수 (f1tenth_env.py:35-36, 로더 f1tenth.py와 동일). NormalizeActions 역식용.
S_MIN, S_MAX, V_MIN, V_MAX = -0.4189, 0.4189, -5.0, 20.0
BASELINE_2LAP = 107.16  # cap-5 step_25k deterministic 2랩 (008 SSOT)


def raw_to_norm(raw, v_max=V_MAX):
    """Diffuser raw action [steer rad, speed m/s] → env NormalizeActions 입력 [-1,1].
    NormalizeActions: original=(a+1)/2*(high-low)+low 의 역. ★v_max=20 고정(per-tier 금지, S4 정정).
    """
    a = np.array([
        2.0 * (raw[0] - S_MIN) / (S_MAX - S_MIN) - 1.0,
        2.0 * (raw[1] - V_MIN) / (v_max - V_MIN) - 1.0,
    ], dtype=np.float32)
    return np.clip(a, -1.0, 1.0)


def env_obs_to_cond(obs, downsample, _ds_fn):
    """f110 obs(lidar 1080 [0,1] + state 5) → 133D conditions 벡터(학습과 동일 구성).
    lidar 다운샘플은 로더의 _downsample_lidar(섹터 min-pool) 그대로 재사용 = train/eval 일치 보장.
    """
    lidar = np.asarray(obs["lidar"], dtype=np.float32).reshape(1, -1)   # (1,1080)
    ld = _ds_fn(lidar, downsample)[0]                                   # (downsample,)
    state = np.asarray(obs["state"], dtype=np.float32)                  # (5,)
    return np.concatenate([ld, state])                                   # (downsample+5,)


def run_episode(policy, env, downsample, ds_fn, batch_size, max_steps, K=1, log_every=500):
    """단일 episode rollout (eval_gate.run_episode 미러, agent만 GuidedPolicy로 교체).

    K-step receding-horizon MPC: 매 재계획마다 GuidedPolicy가 생성한 plan(traj.actions[0],
    horizon개 raw action)의 **앞 K개**를 재계획 없이 순차 실행한 뒤 다시 계획한다. K=1이면
    기존(매 step 재계획)과 동일. K↑는 open-loop 구간을 늘려 K=1 compounding을 완화(단 너무
    크면 open-loop blind, 014 §5-b). --dry(DummyPolicy)는 traj=None이라 K=1로 동작.
    """
    obs = env.reset()
    lap_times, length, cause = [], 0, None
    speeds, steers = [], []   # 진단: 명령 raw speed/steer 분포(D3 점검)
    done = False
    while length < max_steps and not done:
        cond = {0: env_obs_to_cond(obs, downsample, ds_fn)}
        action_raw, traj = policy(cond, batch_size=batch_size, verbose=False)  # raw [steer, speed]
        plan_actions = traj.actions[0] if traj is not None else [action_raw]   # (horizon,2) raw / dry 1개
        for k in range(min(K, len(plan_actions))):
            if length >= max_steps:
                break
            a_raw = plan_actions[k]
            speeds.append(float(a_raw[1])); steers.append(float(a_raw[0]))
            obs, reward, done, info = env.step({"action": raw_to_norm(a_raw)})
            length += 1
            lt = float(obs.get("log_lap_time_s", 0.0))
            if lt > 0.0:
                lap_times.append(lt)
            if length % log_every == 0:
                print(f"    .. step {length} laps={len(lap_times)} cmd_v={float(a_raw[1]):.1f}", flush=True)
            if done:
                cause = info.get("cause")
                break
    return {"cause": cause, "lap_times": lap_times, "length": length,
            "cmd_speed_mean": float(np.mean(speeds)) if speeds else 0.0,
            "cmd_speed_max": float(np.max(speeds)) if speeds else 0.0,
            "cmd_speed_p90": float(np.percentile(speeds, 90)) if speeds else 0.0,
            "cmd_steer_absmean": float(np.mean(np.abs(steers))) if steers else 0.0}


class _DummyPolicy:
    """--dry용: 모델 없이 고정 raw action(직진 8 m/s)으로 env/변환 경로만 검증."""
    def __call__(self, conditions, batch_size=1, verbose=False):
        assert 0 in conditions and conditions[0].shape[0] == int(
            os.environ.get("F1TENTH_LIDAR_DOWNSAMPLE", "128")) + 5
        return np.array([0.0, 8.0], dtype=np.float32), None


def build_policy(scale, batch_size, diff_subpath=None, val_subpath=None):
    """diffusion+value load → GuidedPolicy (GPU). load_diffusion device 기본 cuda:0.

    diff_subpath/val_subpath로 로드 경로 override(기본=P6 원본 그대로, 기존 동작 무변경).
    ★ Track A complete prior 평가 = --diff_subpath diffusion/f1tenth_complete_H128_T20 +
      프로세스 F1TENTH_MODE=complete. 후자가 필수: load_diffusion이 dataset_config로 dataset을
      재생성(serialization.py)할 때 로더 모듈전역 F1TENTH_MODE가 반영돼 normalizer가 complete-stats로
      재fit → complete prior와 정합. (안 주면 all-stats로 fit돼 prior 좌표계 불일치 = P6식 실패.)
      value는 scale=0이면 로드만 되고 무효 → 기존 P6 value 재사용 가능(normalizer 불일치도 ×0 무해).
    """
    from diffuser.utils import load_diffusion, check_compatibility
    from diffuser.sampling import n_step_guided_p_sample, GuidedPolicy, ValueGuide

    H, T, D = 128, 20, 0.99   # config.f1tenth (D5 정정: value discount 0.99)
    diff_lp = diff_subpath or f"diffusion/f1tenth_H{H}_T{T}"
    val_lp = val_subpath or f"values/f1tenth_H{H}_T{T}_d{D}"
    print(f"[plan] load diffusion={diff_lp} value={val_lp} (cwd={os.getcwd()})", flush=True)
    diff_exp = load_diffusion("logs", "f1tenth", diff_lp, epoch="latest")
    val_exp = load_diffusion("logs", "f1tenth", val_lp, epoch="latest")
    check_compatibility(diff_exp, val_exp)
    print(f"[plan] diffusion epoch={diff_exp.epoch} value epoch={val_exp.epoch}", flush=True)
    guide = ValueGuide(val_exp.ema)
    policy = GuidedPolicy(
        guide=guide, diffusion_model=diff_exp.ema, normalizer=diff_exp.dataset.normalizer,
        preprocess_fns=[], sample_fn=n_step_guided_p_sample,
        scale=scale, n_guide_steps=2, t_stopgrad=2, scale_grad_by_std=True,
    )
    return policy


def main():
    ap = argparse.ArgumentParser(description="P5 Diffuser 2랩 평가 (Oschersleben)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="GuidedPolicy는 action[0,0]만 사용 → 1로 충분(속도). 다양성 원하면↑")
    ap.add_argument("--K", type=int, default=1,
                    help="K-step MPC: plan 앞 K개 action 순차 실행 후 재계획(1=매 step 재계획; compounding 완화는 2/3/5)")
    ap.add_argument("--scale", type=float, default=0.1, help="value guidance 강도(config plan 기본; Track A BC=0)")
    ap.add_argument("--max_steps", type=int, default=9000, help="policy step 상한(=time_limit/action_repeat)")
    ap.add_argument("--diff_subpath", default=None,
                    help="diffusion 로드 경로(기본 P6 원본; Track A complete prior=diffusion/f1tenth_complete_H128_T20). "
                         "★complete prior는 F1TENTH_MODE=complete로 실행해야 normalizer 정합.")
    ap.add_argument("--val_subpath", default=None, help="value 로드 경로(기본 P6 value; scale=0이면 무효)")
    ap.add_argument("--out", default="/tmp/p5_eval.json")
    ap.add_argument("--dry", action="store_true", help="모델 미로드 + 더미정책으로 env/변환 CPU 스모크")
    args = ap.parse_args()

    from diffuser.datasets.f1tenth import _downsample_lidar
    from eval_gate import build_config, is_completed
    from dreamer import make_env
    downsample = int(os.environ.get("F1TENTH_LIDAR_DOWNSAMPLE", "128"))

    policy = _DummyPolicy() if args.dry else build_policy(
        args.scale, args.batch_size, args.diff_subpath, args.val_subpath)

    cfg = build_config("f1tenth_Oschersleben")          # device cpu, v_max 20
    assert abs(cfg.v_max - 20.0) < 1e-6, f"★ eval env v_max must be 20 (S4), got {cfg.v_max}"
    env = make_env(cfg, "eval", 0)

    episodes = []
    n = 1 if args.dry else args.episodes
    for i in range(n):
        try:
            res = run_episode(policy, env, downsample, _downsample_lidar,
                              args.batch_size, args.max_steps if not args.dry else 12, K=args.K)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = {"cause": f"ERROR:{type(e).__name__}:{e}", "lap_times": [], "length": 0}
        episodes.append(res)
        two_lap = sum(res["lap_times"]) if is_completed(res["cause"]) and len(res["lap_times"]) >= 2 else None
        print(f"[plan] ep{i+1}/{n} cause={res['cause']} laps={[round(t,2) for t in res['lap_times']]} "
              f"2lap={two_lap} len={res['length']} cmd_v(mean/p90/max)="
              f"{res.get('cmd_speed_mean',0):.1f}/{res.get('cmd_speed_p90',0):.1f}/{res.get('cmd_speed_max',0):.1f} "
              f"steer|{res.get('cmd_steer_absmean',0):.3f}|", flush=True)
    env.close()

    completed = [e for e in episodes if is_completed(e["cause"])]
    two_laps = [sum(e["lap_times"]) for e in completed if len(e["lap_times"]) >= 2]
    agg = {
        "n": len(episodes), "n_completed": len(completed),
        "completion_rate": (len(completed) / len(episodes)) if episodes else 0.0,
        "two_lap_best": float(min(two_laps)) if two_laps else None,
        "two_lap_median": float(np.median(two_laps)) if two_laps else None,
        "baseline_2lap": BASELINE_2LAP,
        "beats_baseline": bool(min(two_laps) < BASELINE_2LAP) if two_laps else False,
    }
    print(f"[plan] AGG {json.dumps(agg, ensure_ascii=False)}", flush=True)
    if not args.dry:
        json.dump({"agg": agg, "episodes": episodes, "scale": args.scale},
                  open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"[plan] saved {args.out}", flush=True)
    print("PLAN_F1TENTH_DRY_OK" if args.dry else "PLAN_F1TENTH_DONE", flush=True)


if __name__ == "__main__":
    main()

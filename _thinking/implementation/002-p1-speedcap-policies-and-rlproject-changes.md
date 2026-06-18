# 002 — P1 구현: 속도캡 behavior policy 학습 + RL_project 코드변경 (세션초기화 핸드오프)

> 2026-06-18. plan_new/008(SSOT v4) 실행 착수(P1). **세션 초기화 직전 핸드오프** — 새 세션이
> 이 문서로 현재 상태·RL_project 코드변경·다음 단계를 다 파악하게 작성. ★ RL_project(별도 프로젝트,
> `/home/dlacksdn/f1tenth_RL_project`)에 한 **코드 수정**은 새 에이전트가 모르니 §4에 전부 기록 +
> 그쪽 `_thinking/implementation/031`에도 남김. append-only. 선행: [[008-diffuser-plan-v4]],
> [[007-diffuser-critique-v3]], [[005-diffuser-venv-and-handoff]].

---

## 0. 한 줄 상태
**P1(behavior policy 제작) 진행 중.** cap-15 정책 학습 완료(불안정→봉우리 스냅샷 `step_105k` 채택, 2랩
~37s, 100% 완주). **cap-5 학습 시작됨**(runs/cap5_oschersleben, 백그라운드). expert는 **폐기**(교수님 확정).
다음=cap-5 결과확인 → cap-10 → P2 수집 → P0 글루.

---

## 1. ★ 교수님 확정 — expert 폐기 (no-expert)
슬라이드(env_setting raws PDF p7-8) + 교수님 답("나와있는 대로 해주세요"):
- 주제1(우리, Offline RL): "**약 100초대 성능의 policy**로 데이터 수집"(expert 언급 없음).
- 주제2(IRL): "**expert** 주행 데이터" 명시.
→ **우리 과제엔 expert 안 씀.** 데이터 = **우리가 만든 속도캡 정책들로만**(전부 우리 정책, no-expert).
008의 expert tier(10%) fallback 발동 → **제거 확정.**

## 2. cap-15 정책 — 학습 완료, 단 불안정
- **V_MAX=15로 warm-load 학습**(stage1 WM + AC fresh, Oschersleben, joint 0.3, lr×0.5). runs/cap15_oschersleben.
- **★ 학습이 진동(oscillate)했다**: eval이 완주(length~1880)↔크래시를 반복. metrics 240k까지 갔으나
  latest.pt는 크래시 골짜기 → **latest 쓰면 안 됨.** 봉우리(완주)는 metrics 120/160/180/210k = **체크포인트
  step_60k/80k/90k/**`step_105k`(최고, return 750). (체크포인트 step = metrics step ÷2, action_repeat.)
- **채택 = `runs/cap15_oschersleben/step_105k.pt`.** 헤드리스 eval 실측(eval_gate, v_max=15, 5ep):
  **완주율 1.000, lap [19.16, 18.1]s → 2랩 ~37.3s, best 18.1s/랩.** A12/A13 PASS.
- **★ 핵심 발견: cap-15 ≈ near-expert(~37s, expert~35s).** Oschersleben 평균 실현속도 ~14.4 m/s라
  **V_MAX=15가 사실상 제약 안 됨** → cap-15는 "느린 baseline"이 아니라 **빠른 tier(천장 ~37s)**.
- 학습은 **정지함**(진동·퇴화 중, 봉우리 확보). watchdog+dreamer 종료.
- 불안정 원인 추정: **joint_replay 0.3(map_easy3 무캡 혼합)**이 WM을 흔드는 것 → 다음 캡에서 joint 제거 실험 고려.

## 3. cap-5 정책 — 학습 시작(진행 중)
- 사용자 결정: cap-15가 너무 빠름 → **V_MAX=5로 ~100초대 느린 behavior policy 제작**(슬라이드 정합).
- **`runs/cap5_oschersleben`, `scripts/cap5_watchdog.sh`로 백그라운드 가동**(warm-load+joint0.3 cap15 동일, V_MAX=5, --steps 300000).
- 예상 2랩 ~100-120s(평균~4.5m/s). 저속이라 완주 쉬울 듯. **새 세션이 결과 확인**(아래 §6 명령).
- ⚠️ cap-5도 진동하면 봉우리 스냅샷 채택(cap-15처럼). joint 제거는 cap-10에서 의도적 실험.

## 4. ★★ RL_project 코드변경 (새 에이전트 필독 — 별도 프로젝트라 모름)
`/home/dlacksdn/f1tenth_RL_project`에 가한 수정. **전부 backward-compatible(기본 v_max=20 → Dreamer 무변경)**,
차량 물리·reward·_STATE_SCALE 무변경. 상세 = RL_project `_thinking/implementation/031`.
- **V_MAX 파라미터화**(속도캡을 위해): `vendor/dreamerv3-torch/configs.yaml`(defaults에 `v_max: 20.0`),
  `dreamer.py`(make_env에서 `F1Tenth(..., v_max=config.v_max)`), `vendor/.../envs/f1tenth.py`(어댑터에
  `v_max` 인자+getstate/setstate+action_space). 검증: 인자없음→action high 20(Dreamer불변), v_max=15→15, pickle왕복 OK.
- **속도캡 = 훈련시 action space V_MAX 제한**(rollout-clamp 아님). NormalizeActions가 [-1,1]→[V_MIN,v_max] 매핑.
- `scripts/cap15_watchdog.sh`, `scripts/cap5_watchdog.sh` 신설(stage2 패턴 + v_max + bracket-pgrep).
- `scripts/watch_drive.py`, `scripts/eval_gate.py`에 **`--v_max` 옵션 추가**(캡 정책 시각화/평가 시 학습값과 일치 필수).
- 이 변경들 **commit됨**(RL_project master). 새 세션은 git log + implementation/031로 확인.

## 5. P2 데이터 다양성 (사용자 우려 반영)
- **deterministic 추론은 데이터가 글자그대로 동일**(eval 5/5 완전동일 실측 — 사용자 직관 옳음).
- 다양성 확보(왜곡 없이): **① 출발 pose 변경**(다른 출발점, 깨끗) **② 자연 stochastic 샘플링**(정책의
  학습된 분포에서 샘플=자연 행동, 왜곡 아님; argmax만 동일데이터 낳음). **③ action noise(인위적 추가)는
  왜곡 우려 → 빼거나 최소화**(사용자 우려). 결론: **pose변경 + 자연 stochastic**으로 충분.

## 6. 명령 모음 (새 세션용)
```bash
cd /home/dlacksdn/f1tenth_RL_project && source .venv/bin/activate
# cap-5 진행 확인
tail -3 runs/cap5_oschersleben/metrics.jsonl; grep -c '"log_completed": 1' runs/cap5_oschersleben/metrics.jsonl
pgrep -af "[d]reamer\.py.*cap5"; ls -t runs/cap5_oschersleben/*.pt | head
# 헤드리스 완주율·랩타임 평가 (캡값 일치 필수)
python scripts/eval_gate.py --ckpt runs/cap5_oschersleben/step_XXk.pt --task f1tenth_Oschersleben --v_max 5 --episodes 5
# 주행 시각화(WSLg 디스플레이 필요)
python scripts/watch_drive.py --logdir runs/cap5_oschersleben --ckpt <peak>.pt --task f1tenth_Oschersleben --v_max 5
# cap-15 채택 정책(확정): runs/cap15_oschersleben/step_105k.pt  (--v_max 15)
```

## 7. P3 주의 (잊지 말 것)
npz의 action은 **tier-상대 정규화**([-1,1]에서 +1이 cap5→5, cap10→10, cap15→15 m/s). Diffuser 데이터셋은
**각 tier v_max로 raw 역정규화해 일관화** 필요 → P2 수집 시 **에피소드별 v_max 기록**.

## 8. 다음 단계
1. **cap-5 결과 확인**(완주율·2랩시간, §6). 봉우리 스냅샷 채택.
2. **cap-10 학습**(V_MAX=10; joint 제거 실험 권장). 중간 tier.
3. **P2 수집**: cap-5/10/15 정책 + pose변경+stochastic으로 rollout, lidar+raw+pose+**v_max(tier)** 기록, tier당 ≥수십 ep, 크래시 폐기.
4. **P0**(병행): 새 .venv에서 Diffuser 글루 디커플([[006-glue-correction-and-data-contract]]) + 1-step.
5. **P3**: f1tenth 로더(§7 역정규화) + normalizer.
- tier 확정(no-expert): {cap-5(느림~100s baseline), cap-10(중간), cap-15(빠름~37s 천장=step_105k)}.
- 분기마다 _thinking + commit + push. push: github.com/dlacksdn/f1tenth_planning_with_diffusion.

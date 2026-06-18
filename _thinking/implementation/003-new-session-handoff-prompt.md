# 003 — 새 세션 부팅 프롬프트 (세션 초기화 핸드오프)

> 2026-06-18. 세션 초기화(clear) 전에, 새 Claude Code 세션을 부팅할 때 그대로 붙여넣을 프롬프트를
> 보존. 내용은 implementation/002(P1 핸드오프 SSOT)를 가리키며, 틀리기 쉬운 핵심을 직접 박았다.
> (프롬프트를 잃어도 새 세션이 MEMORY + implementation/002만 읽으면 동일하게 이어갈 수 있다.) append-only.

---

```text
나는 f1tenth 자율주행 offline RL 프로젝트(Diffuser)를 이어서 진행한다. 이전 세션이 P1(behavior policy
제작) 중간까지 했고 모든 상태를 문서로 남겼다. 너는 (1) 먼저 핸드오프 문서를 읽어 현재 상태를 정확히
파악하고, (2) 진행 중인 cap-5 학습 결과부터 확인한 뒤, (3) 내가 지시하면 다음 단계로 간다. 한국인이니
한글로 답해라. 임의로 _thinking 문서 저장·코드 구현 금지(명시 지시 때만). 추정 말고 1차 소스로 검증해라.

# 먼저 읽을 문서 (SSOT, 순서대로 — 읽기 전에 행동 금지)
- MEMORY는 자동 로드됨. f1tenth-lewm-project 메모리의 "★ P1 진행상황" 줄 = 현재 상태 요약.
- /home/dlacksdn/f1tenth_planning_with_diffusion/_thinking/implementation/002-p1-speedcap-policies-and-rlproject-changes.md
  ← ★ P1 핸드오프 SSOT (현재 상태·RL_project 코드변경·명령·다음단계 전부). 가장 먼저.
- .../plan_new/008-diffuser-plan-v4.md  ← 전체 계획(tier·Phase·제약). 단 expert tier는 폐기됨(아래).
- .../plan_new/007-diffuser-critique-v3.md  ← 008에 대한 직전 적대적 검토(반영됨).
- .../analysis/006-glue-correction-and-data-contract.md  ← P0 Diffuser 글루 디커플 지도.
- .../env_setting/005-diffuser-venv-and-handoff.md  ← 전용 py3.8 venv.
- /home/dlacksdn/f1tenth_RL_project/_thinking/implementation/031-vmax-speedcap-param-for-diffuser.md
  ← ★ RL_project(별도 프로젝트)에 한 코드변경 상세. 새 세션은 이걸 모르니 필독.
(_thinking은 append-only, 명시 저장 요청 때만 저장/전체읽기. 분기마다 문서+commit+push 필수, 자율 git pull 금지.)

# 프로젝트 정체성 / 경로
- 과제(AIE4003 개인, 주제1 Offline RL): 느린 policy로 데이터 수집 → 환경 추가상호작용 없이 더 빠른 정책
  학습. 산출=behavior policy보다 빠른 2랩 lap time + 보고서. 트랙=Oschersleben 단일(map_easy 무관).
- 모델=Diffuser(Planning with Diffusion, Janner ICML2022). clone=/home/dlacksdn/planning_with_diffusion.
- Diffuser 프로젝트 루트=/home/dlacksdn/f1tenth_planning_with_diffusion (전용 venv=.venv, py3.8).
- 데이터 생성원=/home/dlacksdn/f1tenth_RL_project (DreamerV3. 속도캡 정책을 여기서 학습. venv=.venv py3.8).
- 모든 lap time은 2랩 기준으로 말한다(규약).

# 현재 상태 (P1 진행 중 — 자세히는 implementation/002)
- ★ expert 폐기 확정(교수님 "나와있는대로"+슬라이드: expert=주제2 IRL, 우리 주제1=~100s policy no-expert).
  데이터=우리가 만든 속도캡 정책들로만.
- cap-15 학습 완료(runs/cap15_oschersleben): 학습이 진동해서 latest.pt 못 씀. 채택=봉우리 step_105k.pt
  (완주율1.0, 2랩~37.3s). 단 V_MAX=15는 트랙 평균속도(~14.4)보다 높아 사실상 무캡 → cap-15≈near-expert(~37s)
  = 빠른 tier(천장)이지 느린 baseline 아님.
- cap-5 학습 진행 중(runs/cap5_oschersleben, scripts/cap5_watchdog.sh): ~100s 느린 baseline 목표.
  ← ★ 너의 첫 확인 대상.
- tier(no-expert) 목표 = {cap-5 느림~100s baseline / cap-10 중간(TODO) / cap-15 빠름~37s=step_105k}.

# 검증된 사실 (재도출 금지, 단 해석은 가능)
- RL_project 코드변경 commit됨(master fd07ee0): V_MAX 파라미터화(configs.yaml v_max:20.0 / dreamer.py
  make_env v_max=config.v_max / envs/f1tenth.py 어댑터 v_max). 전부 backward-compat=기본20이면 Dreamer 불변
  (실측 검증). 속도캡=훈련시 action space V_MAX 제한, NormalizeActions가 [-1,1]→[V_MIN,v_max] 매핑.
  scripts/cap15_watchdog.sh·cap5_watchdog.sh 신설. watch_drive.py·eval_gate.py에 --v_max 옵션 추가.
- ★ 캡 정책을 watch/eval할 땐 반드시 --v_max를 학습값과 일치(예 cap-5→--v_max 5). 안 하면 정책이
  왜곡되게 빨라짐(NormalizeActions 매핑 어긋남).
- watchdog의 is_alive는 bracket-trick `[d]reamer\.py.*capN` — 너도 프로세스 체크 시 `[d]`/`[c]` bracket
  써서 pgrep cross-match 오탐 방지.
- deterministic 추론은 데이터가 글자그대로 동일(eval 5/5 완전동일 실측). 학습 진동 시 봉우리 스냅샷을 골라 쓴다.

# 절대 제약 / 규약
- 차량/환경 물리·env reward·맵·종료조건·_STATE_SCALE 무변경. 바꾼 건 V_MAX(action bound) 하나뿐.
- Dreamer 본래 학습 기능 무침범(기본 v_max=20). 속도캡 정책 학습은 데이터 수집이라 offline 제약 무위반.
- expert 안 씀. 모든 lap=2랩. 보고서엔 "diffusion 기반 offline planning"으로 정직하게.
- 분기마다 _thinking 문서+commit+push(사전승인). push=Diffuser→github.com/dlacksdn/f1tenth_planning_with_diffusion(main),
  RL_project→f1tenth_RL_project(master). 자율 git pull 금지.

# P2/P3 주의 (잊지 말 것)
- P2 데이터 다양성: 출발 pose 변경 + 정책의 자연 stochastic 샘플링(argmax 아님)으로 확보. action noise
  (인위적 추가)는 왜곡우려로 지양. rollout 시 lidar+raw+pose+★에피소드별 v_max(tier) 기록, 크래시 폐기.
- P3: npz action은 tier-상대 정규화(+1이 cap5→5/cap15→15m/s). Diffuser 데이터셋 만들 때 각 tier v_max로
  raw 역정규화해 일관화 필수(안 하면 action 의미가 tier마다 달라 학습 깨짐).

# 너의 즉시 할 일
1. 위 문서들(특히 implementation/002 + 031) 읽고 현재 상태 파악.
2. cap-5 학습 결과 확인(1차 소스):
   cd /home/dlacksdn/f1tenth_RL_project && source .venv/bin/activate
   tail -3 runs/cap5_oschersleben/metrics.jsonl ; grep -c '"log_completed": 1' runs/cap5_oschersleben/metrics.jsonl
   pgrep -af "[d]reamer\.py.*cap5" ; ls -t runs/cap5_oschersleben/*.pt | head
   # 완주 봉우리 후보를 헤드리스 평가(완주율·2랩 lap):
   python scripts/eval_gate.py --ckpt runs/cap5_oschersleben/step_XXk.pt --task f1tenth_Oschersleben --v_max 5 --episodes 5
3. 결과를 (1)cap-5 완주율·2랩시간 (2)채택 스냅샷 (3)다음 제안(cap-10 학습/P2/P0)으로 간결히 보고만 해라.
   내가 확인하고 지시하면 그때 진행. 임의 저장·구현 금지.
```

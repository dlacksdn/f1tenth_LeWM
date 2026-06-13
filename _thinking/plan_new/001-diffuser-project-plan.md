# 001 — Diffuser offline RL 프로젝트 계획 (v1, 새 라인 시작)

> 2026-06-13. 모델을 LeWM(JEPA, reward-free) → **Diffuser**(value-guided diffusion planner,
> Janner et al. ICML 2022)로 전환하고 새로 시작하는 계획서. 이전 계획(plan_old/)은 LeWM 라인.
> 근거: 코드 분석 [[003-diffuser-code-analysis]], 재사용 [[004-dreamer-reuse-and-behavior-policy]].
> 이 문서가 새 라인의 SSOT. critic 검토 후 구현 진입. append-only.

---

## 배경 (이 문서만 읽어도 되도록)

**과제** (AIE4003 개인 추가과제 1, [[003-project-spec]]): ~100초대 성능의 policy로 주행 데이터를
모으고, **환경 추가 상호작용 없이(offline RL)** 더 빠른 주행 정책을 학습. 산출물 = behavior
policy보다 빠른 lap time을 내는 정책 + 보고서.

**왜 Diffuser인가** (LeWM에서 전환한 이유): LeWM은 reward-free goal-reaching이라 "빠르게"가
목적함수가 아니었고(Δs 간접 손잡이 + 단일프레임 속도 비관측), 렌더링 강제 등 마찰이 컸다.
Diffuser는 ① **offline-native**(과제 필수), ② **value guidance로 return(=빠른 진행)을 직접
최적화**, ③ 네이티브 lidar+state 사용(렌더링·속도비관측 문제 소멸), ④ ICML 2022 landmark·인용
다수(범용성·검증성, 사용자의 Dreamer 선택 기준과 동일)라 선택. Dreamer(생성·RSSM)와 다른
diffusion 패러다임이라 새 경험.

**핵심 동작**: 궤적 diffusion 모델 + value 함수를 offline 데이터로 학습 → 주행 시 현재 관측을
조건으로 미래 궤적을 생성하되 value gradient가 고-return 쪽으로 유도 → 첫 action 실행 → 재생성(MPC).

---

## 목표 (확정)

- **트랙: Oschersleben 단일** (map_easy 건너뜀, 2026-06-13 사용자 지시). centerline 보유.
- **성공 기준**: 최종 Diffuser 정책이 Oschersleben에서 behavior policy(~100초대)보다 빠른 lap
  time(sim 시간 기준, 2랩) + 완주. Dreamer 16.6s는 헤드룸 증거(상한 참고).
- **절대 제약**: f1tenth 차체 물리값 무변경.

---

## 성능이 향상되는가, 천장은 없는가 (핵심 질문 정면 답변)

**향상된다 — 그리고 "behavior policy 속도"에 천장이 고정되지 않는다.** 단 더 높은 곳에
"데이터 커버리지"라는 천장은 있다. 정확히:

- **메커니즘**: Diffuser는 (a) 데이터의 좋은 구간들을 **재조합(stitching)** 하고, (b) **value
  guidance**가 생성을 고-return 궤적으로 민다([[003-diffuser-code-analysis]] §3). 그래서 어떤
  단일 trajectory보다 좋은(빠른) 궤적을 합성 → **behavior policy를 넘어선다.** 이게 Diffuser가
  D4RL에서 behavior policy를 이기는 검증된 원리이고, 과제의 "기존 policy 대비 개선"과 정합.
- **있는 천장 = 데이터 커버리지**: diffusion은 **데이터 분포 안/근처만 생성**한다(이게 offline의
  안전장치 = OOD 환각 억제). 데이터에 **전혀 없는 dynamics는 못 만든다.** 즉 향상 폭은 "데이터에서
  재조합 가능한 최선"까지.
- **그래서 데이터 설계가 향상 폭을 결정한다**: 느린 policy **하나만** 균일하게 모으면 재조합할
  재료가 빈약해 향상이 작다. **느린 policy + 다양성(여러 초기 체크포인트 + action noise로 빠른
  구간/다양한 라인 포함)** 을 모아야 value guidance가 끌어올릴 재료가 생긴다
  ([[004-dreamer-reuse-and-behavior-policy]] §6).
- **요약**: behavior policy(예: 100s) → Diffuser가 그보다 빠른(예: 60~80s대?) 주행 생성은
  구조적으로 가능. 정확한 향상 폭은 데이터 다양성 + Diffuser가 f1tenth 관측을 얼마나 잘
  모델링하느냐에 달림(아래 리스크). "100s 데이터 → 100s밖에 못함"은 **아님**.

---

## 설계 결정 (v1)

### D1. 관측 표현 — lidar 차원 축소 (유일한 실제 리스크)
- raw lidar 1080을 그대로 쓰면 Diffuser가 매 step 1080차원 미래 lidar를 생성해야 해 무겁고
  어려움([[003-diffuser-code-analysis]] §2).
- **v1: 단순 다운샘플 1080 → ~64~108 빔**(stride 또는 max-pool, 구조 보존·학습 0·결정적) +
  state 5 → obs ~70~113차원. transition_dim = action 2 + obs.
- 안 되면 (대안) 학습 encoder 축소(~32) 또는 centerline 피처 기반 state.

### D2. behavior policy — ~100초대, 다양성 포함
- Oschersleben 학습 **초기 체크포인트**(예: step_5k)로 ~100초대 policy 확보. 없으면 warm-load로
  짧게 재학습해 생성([[004-dreamer-reuse-and-behavior-policy]] §6).
- **단일 policy 금지** — 여러 초기/중간 체크포인트 + action noise로 커버리지 확보(향상 재료).

### D3. 데이터 수집 — RL_project 인프라 재사용
- `tools.simulate` + snapshot 로드로 Oschersleben rollout → npz (lidar/state/action/reward).
  파일럿(50 ep) → 본 수집. 차체 물리값 무변경.

### D4. value guidance 대상 reward
- progress 중심(`log_reward_progress`) + 충돌 페널티 유지로 "빠르되 안 박는" return 설계.
  full `reward` vs progress-only는 파일럿에서 비교.

### D5. 학습 대상 2개 + 환경
- ① 궤적 diffusion(train.py 적응) ② value 함수(train_values.py 적응). 둘 다 같은 offline 데이터.
- d4rl/mujoco 의존성 제거, 핵심 torch 코드만 → **RL_project py3.8 venv(torch 2.4) 재사용** 시도
  (1.9→2.4 호환 smoke). 평가도 같은 venv에서 f110_gym과 한 프로세스 → 렌더링·브리지 불필요.

---

## Phase 분해 (게이트 포함)

| # | 작업 | 게이트 |
|---|---|---|
| P0 | 환경: 핵심 Diffuser 코드 py3.8(torch2.4) 구동 smoke (d4rl 디커플) | temporal/diffusion forward 통과 |
| P1 | behavior policy 확보: Oschersleben 초기 체크포인트 or warm-load 재학습 → ~100초대 | lap time ~100s대 + 다양성 확인 |
| P2 | 데이터 수집(파일럿 50ep → 본): rollout + npz | action 재주입 재현, ep 분포 리포트 |
| P3 | lidar 다운샘플 + f1tenth npz 로더(sequence_dataset 포맷) | SequenceDataset 통과, trajectory shape 검산 |
| P4 | 궤적 diffusion 학습 + value 함수 학습 | loss 수렴, 생성 궤적 품질 점검(미래 lidar) |
| P5 | GuidedPolicy를 f1tenth env에 연결(MPC 루프) | 1 ep 완주 시도 |
| P6 | 평가: lap time vs behavior policy | **behavior policy보다 빠름 + 2랩 완주** |

---

## 리스크 (향상 폭을 제한할 수 있는 것)

1. **미래 lidar 생성 난이도** — Diffuser가 (다운샘플) lidar dynamics를 잘 생성 못 하면 value
   입력 궤적이 부정확 → guidance 약화. 완화: 차원 축소(D1), 안 되면 피처 기반 state. P4에서 진단.
2. **value 함수 정확도** — 보상 설계/희소성에 따라 guidance가 약할 수 있음. 완화: progress 중심
   reward(D4), guidance step 수 튜닝.
3. **데이터 커버리지 부족** — behavior policy 다양성이 낮으면 향상 폭 작음. 완화: D2 다양성.

## 다음 단계 (인수인계용)

1. **이 계획을 critic 세션에 검토** (이전 워크플로우와 동일, 결과는 plan_new/002로). 사실 검증
   포인트: lidar 축소가 Diffuser 생성에 충분한가, value reward 선택, 초기 체크포인트 존재 여부.
2. 검토 반영 → P0(환경 smoke)부터 구현.
3. 프로젝트 폴더는 `f1tenth_LeWM` → `f1tenth_planning_with_diffusion` 리네임 예정(사용자).
   clone은 `~/planning_with_diffusion`. 실행 환경 상세는 env_setting/004(일부 갱신 필요).
4. 분기마다 _thinking 문서(analysis/implementation/plan_new) + commit + push.

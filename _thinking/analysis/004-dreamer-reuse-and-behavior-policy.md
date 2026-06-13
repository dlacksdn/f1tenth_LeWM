# 004 — Dreamer(RL_project) 재사용 요소 + behavior policy 생성 전략

> 2026-06-13. Diffuser 전환에 따라 직전 DreamerV3 프로젝트(`~/f1tenth_RL_project`)에서
> 그대로 가져다 쓸 수 있는 요소와, 과제가 요구하는 "~100초대 behavior policy"를 어떻게 만들지
> 정리한 기록. Explore 에이전트의 코드 조사(file:line) 결과를 검증·요약. append-only.

---

## 트랙 결정 (2026-06-13 사용자 지시)

**Oschersleben 단일 트랙.** map_easy는 건너뛴다 — Oschersleben에서만 잘 돌면 됨.
- centerline: `f1tenth_RL_project/maps/Oschersleben_centerline.csv` 이미 존재
- 기존 Oschersleben snapshot: `runs/stage2_oschersleben/policy_best_lap16.6s_step85k.pt`
  (1랩 16.6s = 빠른 expert). **하지만 과제는 "~100초대 policy로 수집"** → 느린 policy를
  따로 만들어야 함 (아래 §6).

---

## 재사용 요소 5가지 (전부 ✅ 가능)

### 1. lidar 인코더 — `dreamer_f1tenth/networks_1d.py:95-242` `ConvEncoder1D`
- 1080빔 → 6-stage Conv1d(stride2) → flatten(128×17=2176) → Linear → **512차원**
- **재사용성**: ⚠️ 부분. 512는 Diffuser 채널로 과대([[003-diffuser-code-analysis]] §2). v1은
  **단순 다운샘플(1080→~64~108)** 우선, 학습 encoder는 필요 시 더 작게 재학습. 구조 참고용으로 유용.

### 2. 데이터 수집(rollout) 인프라 — `vendor/dreamerv3-torch/tools.py:128-200` + `snapshot_utils.py:25-36`
- `simulate(agent, envs, cache, ...)`가 policy 구동 → 에피소드별 npz 저장.
  npz 키: lidar(T,1080) / state(T,5) / action(T,2 raw) / reward / discount / is_* / log_*
- **재사용성**: ✅ 높음. **이 인프라로 Diffuser offline 데이터셋을 그대로 수집**. policy 로드는
  `inference_state_dict`(`_wm.*`+`actor.*`) 형식 — Dreamer policy 전용이라 우리 수집엔 그대로 OK
  (수집 정책 = Dreamer니까).

### 3. warm-load(world model 이식 + actor-critic 초기화) — `vendor/dreamerv3-torch/stage2_utils.py:14-29` + `dreamer.py:301-387`
- `extract_warm_state`: `_wm.*`만 추출 → actor/critic/optimizer는 fresh 초기화.
- **재사용성**: ✅ behavior policy 생성에 직접 사용(§6).

### 4. reward 정의 — `dreamer_f1tenth/envs/f1tenth_env.py:83-470`
- `reward = progress + collision + reverse + diverged + lap`
  (progress = ALPHA·clip(arclen_delta,0,0.5), 종료 페널티 -10, lap 보너스 Oschersleben=100)
- npz에 `reward`(합산)와 `log_reward_progress` 등 성분 별도 저장.
- **재사용성**: ✅ 높음. Diffuser **value 함수가 이 reward로 할인 return 계산**
  ([[003-diffuser-code-analysis]] §3). progress 중심 reward라 "빠른 진행=고 value"가 자연 성립.
- ⚠️ 주의: value 학습 시 어떤 reward를 쓸지 결정 필요 — 전체 `reward`(충돌/lap 포함) vs
  `log_reward_progress`만. lap time 최적화엔 progress가 핵심, 충돌 회피는 페널티로 유지가 합리적.

### 5. env wrapper / obs·action 스펙 — `dreamer_f1tenth/envs/f1tenth_env.py:119-470`
- obs: lidar(1080, [0,1] 정규화) + state(5, `_STATE_SCALE`로 정규화) =
  [vel_x, vel_y, ang_z, prev_steer, prev_speed]
- action: Box([S_MIN,V_MIN],[S_MAX,V_MAX]) = steer[-0.4189,0.4189]rad, speed[-5,20]m/s
- env step 0.02s(sim 0.01×action_repeat 2), gymnasium 5-tuple API,
  `reset(options={"pose":(x,y,θ)})` 지원, info에 `arclen_s`/`closest_idx` 제공
- 종료 우선순위: diverged > collision > reverse > lap_complete(2랩) > timeout(9000step)
- **재사용성**: ✅ 높음. 수집·평가 양쪽에서 동일 env. **차체 물리값 무변경 원칙 유지**(사용자 절대 제약).

---

## 6. behavior policy 생성 전략 (사용자가 강조한 핵심)

**사용자 지적**: Dreamer는 map_easy world model을 가져오고 actor-critic을 초기화해 Oschersleben에서
재학습한 구조다. 즉 **초기 policy부터 학습해서 우리가 원하는 성능대의 policy를 만들 수 있다.**

→ 과제는 "**~100초대 policy로 데이터 수집**"을 요구. 현재 가진 16.6s는 너무 빠름. 해결:

- **방법 A (권장): 학습 초기 체크포인트 사용.** Oschersleben 재학습 곡선의 이른 step(예: step_5k)
  스냅샷이 ~100초대 느린 policy. `snapshot_utils.save_interval_snapshot`이 interval 스냅샷을
  이미 저장(dreamer.py eval_every). 곡선 초반에서 원하는 lap-time대 policy를 골라 쓰면 됨.
  → 만약 그 시점 스냅샷이 디스크에 없으면, warm-load(world model 이식)로 **짧게 재학습**해
  초기~중간 단계 policy를 새로 뽑으면 됨(빠름).
- **방법 B: 의도적 약화.** 빠른 policy의 speed action을 캡(예: ×0.4)해 ~100초대로 — 단 dynamics
  분포가 바뀌므로 A 선호.

**왜 느린 policy가 맞나** (이미 정리됨, [[f1tenth-lewm-project]] 논의): expert(16.6s) 데이터로는
"개선"을 보이려면 expert를 이겨야 해 어려움. **~100초대 behavior policy → Diffuser가 그보다
빠른 주행 생성**이 과제의 "기존 policy 대비 개선"을 가장 명확히 demonstrate. Dreamer가 16.6s까지
가능함을 알기에 100초→개선의 헤드룸은 충분.

⚠️ **데이터 커버리지 주의**: Diffuser는 데이터에 있는 구간을 재조합(stitching)해 개선하므로,
behavior policy가 **단일 속도로만** 달리면 개선 여지가 좁다. 느린 policy 1개만 쓰지 말고
**약간의 다양성(여러 초기 체크포인트 + action noise)** 으로 빠른 구간/다양한 라인을 데이터에
포함시켜야 value guidance가 끌어올릴 재료가 생김. (구체 비율은 plan_new에서 결정)

---

## 7. 다음 분기

- behavior policy 후보 확보: Oschersleben 학습 초기 스냅샷 디스크 존재 여부 확인 → 없으면 warm-load
  짧은 재학습으로 ~100초대 policy 생성
- lidar 다운샘플 스펙 결정(빔 수, max-pool vs stride)
- value 학습용 reward 선택(full reward vs progress)
- 위는 plan_new/001 계획서에서 통합 결정

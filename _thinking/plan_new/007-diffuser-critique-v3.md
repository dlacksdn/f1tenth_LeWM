# 007 — 006(diffuser-plan-v3) 적대적 검토 결과 (critic v3)

> 2026-06-18. [[006-diffuser-plan-v3]](현 SSOT)의 **read-only 적대적 검토**. 특히 006 D2가 (A)rollout-clamp에서
> (B)"속도상한을 걸고 warm-load 재학습"으로 갈아엎인 개정판 — 그 신규 (B)를 1차 소스로 공격 검증.
> 1차 소스 직접 검증: `dreamer_f1tenth/envs/f1tenth_env.py`·`vendor/dreamerv3-torch/envs/f1tenth.py`(adapter)·
> `vendor/dreamerv3-torch/envs/wrappers.py`·`vendor/dreamerv3-torch/dreamer.py`·`configs.yaml`·
> stage1/stage2 `metrics.jsonl`·완주 npz 실측(venv)·체크포인트 인벤토리.
> **어떤 파일도 수정하지 않음. append-only.** 005([[005-diffuser-critique-v2]])가 이 방식의 모범.
> 선행: [[006-diffuser-plan-v3]], [[005-diffuser-critique-v2]], [[006-glue-correction-and-data-contract]],
> [[004-diffuser-plan-v2]], [[004-dreamer-reuse-and-behavior-policy]], [[003-project-spec]] + PDF.

---

## 0. 종합 평결: **수정 후 진행 (Revise-then-proceed)**

006의 (B)"속도상한-훈련" 가설은 **"005가 반증한 건 *무제한* 재학습뿐이고 속도상한 훈련은 미검증이지 반증이
아니다"라는 화해 논리가 타당**하다 — 1차 소스로 지지된다. (B)는 논리적으로 건전하고(상한→고속점프 불가→그
속도대 완주로 수렴), 캡 값 10/15도 실측 속도분포상 **expert(~33s)·중간(~44s)·느림(~62s) 2랩 스펙트럼**을
그럴듯하게 만든다. 005의 6개 지적(P0 확장·디커플·normalizer 정정·피처 parity·baseline·라인순서)도 **거의 다
닫혔다.** 그러나 두 가지가 블로커급:

- **① 006은 더 싸고 깨끗한 대안 (A)rollout-clamp를 *실측 없이 단언으로 폐기*했다.** (A)는 훈련 0·즉시·offline
  제약 적합성이 (B)보다 우수. (B)는 캡당 *fresh actor-critic 풀 재학습*(stage2 실측 첫 완주 ~5.5h, 전체 ~7.7h)
  이라 2회면 ~10–16h — **개인 과제 규모 대비 과投 위험.**
- **② "라인 다양성→expert 초과(Stretch)" 메커니즘은 데이터 구조상 거의 불성립.** 속도상한 정책은 expert보다
  *모든 구간이 느림* → stitching이 expert를 넘을 sub-16.6s 재료가 데이터에 없음.

**결론**: (B)를 **escalation으로 유지하되**, 그 앞에 **30분짜리 (A) 탐침 게이트**를 박고(P1-pre), 천장 서사를
"cap 초과(floor)"로 낮추며, V_MAX 구현 현실(하드코딩 상수)을 명시하면 진행 가능. (B) 자체를 반려하진 않는다.

---

## 1. 검증 방법론 (1차 소스)

| 소스 | 경로 | 무엇을 봤나 |
|---|---|---|
| action space / V_MAX | `dreamer_f1tenth/envs/f1tenth_env.py:36,50,124-132,182-186,306-307` | V_MAX=상수, __init__ 인자 부재, _STATE_SCALE 하드코딩 |
| env 생성 경로 | `vendor/dreamerv3-torch/dreamer.py:204-208` + `envs/f1tenth.py:39,51,95` | F1Tenth(task,action_repeat,seed) — 캡 인자 없음 |
| NormalizeActions | `vendor/dreamerv3-torch/envs/wrappers.py:35-39` | action_space.low/high 동적 매핑 → 캡 상수 변경 시 반영 |
| warm-load 정체성 | `scripts/stage2_watchdog.sh:28,73-77` | stage2 = 무제한 V_MAX=20 warm-load 런 |
| 학습 config | `vendor/dreamerv3-torch/configs.yaml` f1tenth | steps 5e5, time_limit 18000→9000(180s), action_repeat 2 |
| 학습곡선 | `runs/stage2_oschersleben/metrics.jsonl`(671rec), `runs/stage1_map_easy3/metrics.jsonl` | 첫 완주 시점·완주 lap time·eval_length 분포 |
| 완주 실물 | `runs/stage2_oschersleben/train_eps/*.npz`(265) venv 실측 | 실현/명령 속도, 완주 vs 크래시 속도 |
| 체크포인트 | `runs/stage2_oschersleben/*.pt` + `KEEP/`, `runs/stage1_map_easy3/*.pt` | 완주 정책 인벤토리, lap17.4s 위치 |

---

## 2. 사실 검증 (006 주장별 ✅/⚠️/❌)

### (B) 중심 가설 — ★ 핵심

**(B-화해) "005는 무제한 런만 봤다, 속도상한은 미검증이지 반증 아니다"** (006:58-62)
- ✅ **타당.** stage2 = 무제한 V_MAX=20 warm-load 런이 맞고([stage2_watchdog.sh:28,73-77], 캡 인자 없음),
  곡선은 크래시→고속완주 점프다. 속도상한 런은 실행된 적 없음 → 반증 대상 아님. 화해 논리는 1차 소스와 정합.

**(B-a) "속도상한 = env action space V_MAX 제한, 차체 물리 무변경, 설정/래퍼 수준"** (006:70-72)
- ⚠️ **절반만 맞다.** 물리 무변경 ✅(action bound이지 동역학 아님). **그러나 "설정/래퍼 수준"은 사실과 다르다**:
  V_MAX는 **하드코딩 모듈 상수** [f1tenth_env.py:36] `V_MIN, V_MAX = -5.0, 20.0`, `__init__` 시그니처
  [f1tenth_env.py:124-132]에 **V_MAX 파라미터 없음**. env 생성 [dreamer.py:204-208]은
  `F1Tenth(task, action_repeat, seed)`만, adapter [envs/f1tenth.py:51] `__init__(self, task, action_repeat=2, seed=0)`도
  캡 인자 없음. → **config/CLI 경로 전무.** 캡을 걸려면 *코드 편집* 필수.
  - NormalizeActions [wrappers.py:35-39]가 `action_space.low/high`를 **동적**으로 읽어 [-1,1]→[low,high] 매핑하므로,
    상수만 바꾸면 명령 매핑 반영됨 ✅. adapter도 같은 상수 import [envs/f1tenth.py:39,95] → 단일 소스 ✅.
  - **숨은 seam**: [f1tenth_env.py:50] `_STATE_SCALE=[20.0,...,20.0]`은 V_MAX와 무관하게 20 하드코딩
    (vel_x·prev_speed 정규화). 캡을 10으로 내려도 안 바뀌어 속도 피처가 [-0.5,0.5] 압축 범위 → warm WM 분포와
    미세 불일치(양성 추정이나 미명시).
- ⚠️ **per-run 미파라미터화**: cap-10·cap-15를 **상수 손편집**해 따로 돌려야 함(또는 config flag 신설). 006은
  구현 공수를 0으로 취급.

**(B-b) "cap-10 정책이 2랩 완주를 *학습*하나"** (006:67-69)
- ⚠️ **미검증이나 근거상 likely-OK.** warm WM은 ≤10 m/s 영역 데이터를 풍부히 봤음(크래시 ep 평균속도 **10.6 m/s**
  실측, range 2.6~14.6) → WM 동역학 신뢰 가능. 저속이면 코너 여유↑ → 완주 난이도↓. "cap-10 완주 학습"은 합리적.
  **단 실측 0이므로 P1 전 단정 불가**(006도 ⚠️로 표기 — 정직).

**(B-c) "학습 시간이 합리적인가"** (006:85 리스크 #1)
- ❌ **006이 과소평가.** warm-load는 **WM만 이식, actor/critic fresh**(005 §2가 [dreamer.py:300,368-383]로 확인,
  인벤토리 확정). 따라서 각 캡 = *정책 처음부터 재학습*. stage2 실측: tfevents span **27,587s ≈ 7.66h**,
  **첫 완주가 metrics-step 129,280**(전체 178,000의 ~72% ≈ 5.5h). → 캡 1개 ≈ 5~8h, **cap-10+cap-15 ≈ ~10–16h**,
  캡/리워드 조정 재실행 시 배수. 리스크 #1("저속은 보통 완주 쉬움")은 이 multi-hour 비용을 숨김.

**(B-d) "점프 불가→느린완주 수렴"이 논리 비약인가**
- ✅ **비약 아님(건전).** 완주 6개 실측: **명령속도 max가 전부 20.0(천장)**, 평균 ~16; 실현속도 평균 14.4.
  완주는 천장을 적극 사용해 달성된다. 천장을 10으로 막으면 progress 최대화 경로 = "≤10에서 깨끗이 완주" → 느린 완주.
  저속 특유 새 실패모드의 강한 증거 없음(stage1 완주 159회 robust). **(B) 중 가장 방어 가능한 논리.**

**캡 값 10/15의 타당성 (006 미명시 — 실측 보강)**
- ✅ **의외로 합리적.** Oschersleben L_track=275.18m [f1tenth_env.py:76], expert 완주 실현 평균 14.4 m/s
  (명령 평균 16, max 20). 추정 2랩: cap-20(현재) ~33–38s → cap-15 ~44–46s → cap-10 ~60–68s. "60→30" 서사와 정합.
  **단 실현속도≠명령속도라 정확 lap time은 P1 실측까지 미확정**(006이 P1 게이트로 둠 — OK).

### beat-expert / 라인 다양성 (006 Stretch, 006:75-76,108-110)
- ❌ **메커니즘 결함.** "속도별 별개 정책=라인 다양성→상보 stitching으로 expert 초과"는 **데이터 구조상 거의 불성립**:
  속도상한 정책은 정의상 expert보다 *모든 구간이 느림*(캡<expert 평균속도). stitching은 데이터 내 *구간*을
  재조합할 뿐 새 속도를 만들지 못함 → 최선이 "expert 구간 재현 = expert 동급". expert를 넘으려면 sub-16.6s/lap
  구간이 데이터에 있어야 하나 **어디에도 없음**(유일 고속재료 = expert 10% 자체). 006이 "보장 아님"으로 hedge한
  건 ✅이나, 제시한 *메커니즘*(라인 다양성)이 속도 향상을 못 낳는다는 점은 미인지.
  → **Stretch는 "불확실"이 아니라 "본 데이터 구성에선 거의 불가".**

### 005 지적 반영 여부 (point 3) — 거의 다 닫힘
- ✅ **P0 게이트 확장**: [006:128] (A)TemporalUnet import (B)f1tenth 로더 적재 (C)train.py 1-step
  (D)train_values/plan_guided 파서 + Config pickle→Renderer 확인 — 005 §4-#7 그대로 닫힘.
- ✅ **디커플 목록**: [006:91-94] colab.py:17·preprocessing.py:7·sequence.py:7·d4rl.py:25·buffer.py:12 — 005 §3 반영.
- ✅ **normalizer 정정**: [006:95-96] "Trainer.save는 normalizer 미저장, P3 라운드트립" — 005 §4-#5 정확 반영.
- ✅ **online 피처 parity**: [006:97] 단일 피처함수+golden test. (v1=lidar 다운샘플이라 [006:53] 피처 seam은 v2 이연 — 합리적.)
- ✅ **baseline 게이트**: [006:129] "baseline=cap-15 시간 정의, 미달 시 캡/하이퍼 조정". 005 §5 충족.

### offline 제약 해석 (point 6, 006:71-72)
- ❌ **(B)가 가장 취약.** 과제 "환경과의 추가 상호작용 없이 정책 개선"(주제1). *데이터 수집*은 허용되나, (B)는
  데이터를 만들기 위해 **새 정책 2개를 online RL로 ~10–16h 학습** — "데이터 수집"보다 "추가 정책 학습"에 가까워
  방어 곤란. 반면 (A)rollout-clamp는 *기존 정책을 rollout 시 클램프*만 — 새 학습 0이라 offline 서사에 더 깨끗.
  **(B)는 비용·offline 적합성 양면에서 (A)보다 열위.**

### 데이터 구성 (point 4, 006:77-78)
- ⚠️ **expert 10%는 얇음.** value가 고-return(expert)을 선택하려면 expert 노출이 충분해야 하나 10%면 D4RL
  medium-expert 표준 대비 expert-light. 천장 재료 희박 → P4 생성품질 점검 필수(006 인지). 45/45/10은 실측 근거
  없는 잠정치(006도 "조정 가능"). baseline=cap-15는 측정 가능·일관 ✅.
- ⚠️ **완주-only 데이터 절대량**: 크래시 배제 시 stage2 기존 완주 ep는 **단 6개**(실측). (A)든 (B)든 정책을
  **다수 재rollout**해 데이터량 확보 필요 — 006 P2가 "분포 리포트"만 두고 목표 ep 수 없음.

### 확정 인벤토리 교차검증 (사실 정합)
- ✅ **lap17.4s_step80k 존재**: `runs/stage2_oschersleben/KEEP/KEEP_oscher_best_lap17.4s_step80k.pt`. 006:44-45 정확
  (KEEP 디렉토리에 보관). 본진엔 `policy_best_lap16.6s_step82k/85k`.
- ✅ **"느리게 완주하는 체크포인트 없음"**: eval_length>2000(=40s+) = stage2 0/19, stage1 0/52. 완주 lap_time도
  stage2 35.5–37.6s/2랩, stage1 ~28s/2랩이 최저. **60–100s 완주대 전무 — 1차 소스 확정.**
- ✅ **"38s 정책 = replay 에피소드"**: 완주 6 ep는 train_eps npz(replay)이고 35.5–37.6s/2랩. 저장 정책 체크포인트
  아님. 006:46 정확.
- ✅ **stage1은 완주 스펙트럼 체크포인트 보유**(lap6.1/7.0/8.0/9.0/10.3/13.1/15.2/16.2/17.1/18.4s) — 전부 빠름,
  최저 18.4s. "느린 완주는 자연 산출 안 됨" 보강.

---

## 3. 핵심 결함 Top 6 (심각도 + 완화)

| # | 결함 | 심각도 | 완화책 |
|---|------|--------|--------|
| 1 | **(A)rollout-clamp를 실측 없이 폐기 → (B) 2회 풀-재학습(~10–16h) 과投.** (A)는 훈련0·즉시·offline 적합. [006:84] "미스매치로 품질저하"는 단언(정책은 closed-loop 반응형이라 클램프해도 재계획). | **Blocker(process)** | **P1 앞에 (A) 탐침 게이트 신설**: 기존 best를 명령속도 {10,15}로 클램프해 5ep rollout → 깨끗이 완주 + 궤적 품질 OK면 **(A) 채택(무료)**; 가시적으로 나쁠 때만 (B) escalate. "안 되면 그때만" 그대로. |
| 2 | **"라인 다양성→expert 초과" 메커니즘 불성립**(느린 정책은 expert보다 전 구간 느림 → 넘을 재료 부재). | **Major** | Stretch를 **"cap 초과(floor)"로 확정**. expert 초과는 데이터에 sub-16.6s 재료 없는 한 불가 명시. 정말 노리면 expert 비율↑(균일 스케일 아님). |
| 3 | **V_MAX가 config 아닌 하드코딩 상수**(f1tenth_env.py:36), per-run 미파라미터화, `_STATE_SCALE` 20 하드코딩 seam. 구현 공수 0 취급. | **Major** | P1 전 구현 명세: ① V_MAX를 `__init__` 인자/config flag로 승격(상수 손편집·per-run 재편집 금지) ② cap 변경 시 `_STATE_SCALE` 유지/변경 결정 명문화(WM 분포 일관). |
| 4 | **훈련 비용·wall-clock 미정량** — 캡당 fresh actor-critic ~5.5h(첫 완주)/~7.7h(전체) 실측. | **Major** | 리스크 #1에 "캡당 ~5–8h, 2캡 ~10–16h, 조정 재실행 배수" 명기. cap-10 완주 안정화 시 **조기 종료**(500k 다 안 돌림). |
| 5 | **완주-only 데이터 절대량 미정의**(기존 완주 6ep). 수집 목표 ep 수 없음. | **Minor-Major** | P2에 "tier별 최소 완주 ≥N ep(예 30/tier)" 수치 게이트. |
| 6 | **step 카운터 모호**(첫 완주 metrics-step 129k vs 체크포인트명 step85k — action_repeat 분할 등). P1 "학습시간 합리적" 기준 불명. | **Minor** | P1 게이트에 "어느 카운터 + wall-clock 상한(예 8h/캡)" 명시. |

---

## 4. 누락 단계 / 게이트 (007에 신설)

1. **[P1-pre (A) 탐침 게이트]** 기존 best 명령속도 클램프 rollout(코드 트리비얼, 004 D3 `SpeedCappedAgent` 재활용)
   → 완주·품질 통과 시 (B) 스킵. **최우선 누락.**
2. **[V_MAX 주입 구현 게이트]** 캡을 `__init__`/config로 배선 + cap-10/cap-15 두 값 비손편집 전환 확인.
3. **[P2 데이터량 게이트]** tier별 최소 완주 ep 수 + 완주율 리포트(현재 "분포 리포트"만).
4. **[P1 wall-clock 상한]** 캡당 학습 시간 budget + 조기 종료 기준(완주 안정화 시 stop).
5. **[P4 천장 점검]** expert 10%로 value가 고-return을 실제 선택하는지(생성 궤적이 expert 구간 활용하는지) 조기
   판정 → 부족 시 비율 상향(006 인지, 게이트화 권장).

---

## 5. 확인 못 한 항목 (침묵 금지)

1. **cap-10/15 정책이 실제로 2랩 완주하는지** — 미실행(P1). 근거상 likely-OK이나 단정 불가.
2. **클램프(A) 데이터 품질이 정말 (B)보다 나쁜지** — 둘 다 미rollout. 006의 (A) 폐기 근거는 미검증 단언.
3. **다른 V_MAX 정책이 실제로 다른 라인을 학습하는지** — 데이터 없음(투기). 단 속도향상엔 무관(§2 메커니즘 결함 우선).
4. **cap 적용이 학습을 단축/연장하는지** — 미검증(저속이 더 빨리 완주할 수도, progress 희소로 더 늦을 수도).
5. **실현속도→정확 lap time 환산** — 명령≠실현(완주 ep 명령16/실현14.4). cap별 lap은 P1 실측 전 ±수초 불확실.
6. **`_STATE_SCALE`/20 압축이 warm WM 적응에 미치는 영향** — 양성 추정, 미검증.
7. **글루 import 체인(rendering/d4rl/colab) 실 통과** — 005 정적 확인, 본 검토는 (B)에 집중해 재실행 안 함(P0 smoke).
8. **stage2 step 카운터 정합**(metrics 0~178k vs 체크포인트 step85k) — action_repeat 분할 추정, 미정밀화.

---

## 6. 006 → 007(plan-v4) 수정안 diff 요약

```
[§배경/D2 헤더]  (B) 유지하되 위상 강등
- "확정 설계: 느림·중간 tier = 속도상한 warm-load 재학습"
+ "주경로 = (A) 기존 best 명령속도 클램프 rollout(훈련0). (B)속도상한-재학습은
+  (A) 데이터 품질이 P1-pre 탐침에서 가시적으로 불량할 때만 escalate."

[D2 — (A) 폐기 근거 정정]  ★ Blocker
- [006:84] "rollout-clamp = 미사용. 고속 조향 저속 강제 미스매치로 품질저하"
+ "(A) 미스매치는 *미검증 단언*. 정책은 closed-loop 반응형 → 클램프해도 관측 기반
+  재계획. P1-pre에서 5ep 실측 후 판정. (B)는 캡당 fresh actor-critic 풀 재학습
+  (stage2 실측 ~5.5h 첫완주/~7.7h)이라 2캡 ~10–16h = 과投 위험."

[성능표 — Stretch 메커니즘 정정]
- "Stretch: best 초과 — 속도별 정책=라인 다양성→상보 stitching, 가능성 있음"
+ "Stretch 강등: 속도상한 정책은 expert보다 전 구간 느림 → stitching이 넘을 sub-16.6s
+  재료 부재. Floor=cap-15 초과로 확정. expert 초과는 본 데이터 구성상 불가."

[D3 — V_MAX 구현 현실 명시]
+ "V_MAX는 config 아닌 모듈 상수 [f1tenth_env.py:36], __init__ 인자 없음 [:124-132],
+  adapter도 동일 [envs/f1tenth.py:51]. → P1 전 V_MAX를 __init__/config로 승격
+  (상수 손편집·per-run 재편집 금지). _STATE_SCALE[:50]=20 하드코딩 유지/변경 결정 명문화."

[리스크 #1 — 비용 정량화]
+ "캡당 ~5–8h(fresh actor-critic), 2캡 ~10–16h, 조정 재실행 배수. cap-10 완주 안정화 시 조기 종료."

[Phase — 게이트 신설]
+ P1-pre: (A) 클램프 탐침(5ep/캡) → 완주·품질 OK면 (B) 스킵.
+ P1: V_MAX 주입 구현 + wall-clock 상한(8h/캡) + 카운터 명시.
+ P2: tier별 최소 완주 ep 수치(예 ≥30/tier).

[offline 서사]
+ "(A) 채택 시 'expert를 속도클램프해 느린 behavior policy 생성' = 추가 학습0이라
+  offline 제약에 (B)보다 정합. 보고서에 이 점 강조."
```

---

## 7. 다음 단계 (인수인계)

1. **이 문서 = 006 D2의 (B) 검증 입력.** 다음 계획 에이전트는 §6 diff를 반영해 plan-v4 작성하되, **(A) 탐침을
   P1-pre 게이트로 선행**시키고 (B)는 escalation으로 강등.
2. **P0는 (A)/(B)와 무관하게 즉시 선행 가능**(글루 디커플 + 1-step + 파서). 006 P0 게이트 그대로 유효.
3. (A) 탐침이 통과하면 V_MAX 구현(결함 #3)·2캡 풀-재학습(결함 #4) 비용을 통째로 절약.
4. 분기마다 _thinking 문서 + commit + push(상시 규약, 자율 pull 금지). f1tenth 판단 시 [[003-project-spec]] + PDF.

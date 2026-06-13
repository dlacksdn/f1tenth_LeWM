# 005 — 004(diffuser-plan-v2) 적대적 검토 결과 (critic v2)

> 2026-06-14. [[004-diffuser-plan-v2]](현 SSOT)에 대한 **read-only 적대적 검토**.
> 1차 소스(RL_project의 `metrics.jsonl`·265개 `train_eps/*.npz`·스냅샷·학습 스크립트,
> Diffuser 코드 `~/planning_with_diffusion`, 새 py3.8 venv import 실측)를 직접 열어 사실 검증.
> **어떤 파일도 수정하지 않음. append-only.** 002([[002-diffuser-critique]])가 이 방식의 모범.
> **목적: 004의 D2를 재설계할 다음 계획 에이전트(plan_new/006 또는 v3)의 입력 문서.**
> 선행: [[004-diffuser-plan-v2]], [[002-diffuser-critique]], [[006-glue-correction-and-data-contract]],
> [[005-diffuser-venv-and-handoff]], [[004-dreamer-reuse-and-behavior-policy]], [[003-diffuser-code-analysis]].

---

## 0. 종합 평결: **수정 후 진행 (Revise-then-proceed)** — 단, **D2는 진행 전 필수 재작성**

모델(Diffuser)·코드 적용성·전용 venv·데이터 계약은 002 이후로 **더 단단해졌고 건전하다**(D1/D4/D5는
유지+정밀화면 충분). 그러나 **004의 심장인 D2(데이터소스 = "Dreamer warm-load 재학습"을 *주경로*로)는
그 핵심 전제가 1차 소스로 반증된다.**

004:59는 *"002 §4의 '느린 완주 스냅샷 부재'는 재학습으로 무력화된다"*고 단언한다. 그러나 검증 결과:

- **(a)** 그 재학습 실험은 **이미 실행됐다 — 산출물이 바로 `runs/stage2_oschersleben`이다.**
  (`stage2_watchdog.sh`: `WARM_CKPT=stage1_map_easy3/latest.pt`, `--warm_load_ckpt`.)
- **(b)** 그 학습곡선은 **[크래시(259/265 = 97.7%)] → [고속 완주(~17–19s)]로 점프**하며,
  그 사이 **"느리지만 완주(40–80s)"하는 구간이 존재하지 않는다.**
- **(c)** warm-load 소스인 **stage1조차** 완주는 13–27s뿐, **60–80s를 만든 적이 없다.**

즉 **D2는 "이미 알려진 실패를 P1에서 다시 재현하라"는 계획**이다. 게다가 향상의 재료인
"빠르되 안 박는(fast-and-surviving)" 구간이 데이터에 거의 부재(고속+고잔여진행 32-step 윈도 = **0.04%**)해,
value guidance(=할인 return-to-go)가 오히려 "느리고 안전한" 쪽을 선호할 구조다 → 향상 천장이 데이터에
의해 낮게 고정된다.

**결론**: 모델·코드 라인은 그대로 가되, **D2를 002 §3–4의 결론("느린 완주 baseline을 speed-cap으로
*제작*")으로 되돌리거나, 최소한 P1에 즉시-폴백 게이트를 박아야** 한다. 이 한 가지가 블로커다.
나머지는 누락 게이트 보강(§5)으로 해결된다.

---

## 1. 검증 방법론 (다음 에이전트가 재확인할 1차 소스)

| 소스 | 경로 | 무엇을 봤나 |
|---|---|---|
| stage2 학습곡선 | `RL_project/runs/stage2_oschersleben/metrics.jsonl` | eval_length/완주/lap time 추이 |
| stage1 학습곡선 | `RL_project/runs/stage1_map_easy3/metrics.jsonl` | 완주 분포(중간 느린대 유무) |
| offline 데이터 실물 | `RL_project/runs/stage2_oschersleben/train_eps/*.npz` (265개) | 속도/진행/완주/return-to-go 분포 |
| warm-load 정체성 | `RL_project/scripts/stage2_watchdog.sh`, `vendor/dreamerv3-torch/dreamer.py:300,368-383` | stage2가 warm-load 런인지 |
| Diffuser 코어/글루 | `~/planning_with_diffusion/diffuser/**` | import 체인·normalizer·value·Markov |
| 새 venv import | `f1tenth_planning_with_diffusion/.venv/bin/python` 실런 | P0 디커플 실측 |

---

## 2. ★ 결정적 사실 — `stage2_oschersleben`가 곧 004 D2가 제안하는 실험이다

- `stage2_watchdog.sh`(L1–9, 28, 75–76): `WARM_CKPT`가 **`stage1_map_easy3/latest.pt`**를 가리키고
  `--warm_load_ckpt`로 전달 → stage2는 **"stage1 world model warm-load → Oschersleben 재학습" 런**.
- `dreamer.py:300, 368–383`(`_do_warm`): **`_wm.*` 키만 로드**, actor/critic은 fresh 초기화 →
  004:56–57의 설계("world model만 warm-load(actor-critic 초기화)")와 **정확히 일치**.
- `stage1_map_easy3/latest.pt` **존재**(148MB). 즉 **파일이 제약이 아니라, 그것이 만드는 데이터가 제약**.

→ 004 D2가 "이제 하자"는 실험은 **셋업까지 동일하게 이미 끝났고**, 그 결과 곡선이 002 §4가 분석하고
이번에 재확인한 `metrics.jsonl`이다. **"재학습으로 무력화"는 자기 데이터로 반증된다.**

---

## 3. 사실 검증 (004 주장별 ✅/⚠️/❌)

### D2 — 데이터소스 (★ 핵심)
- ❌ **"warm-load 재학습이 60–80s 완주 policy를 만든다"** (004:52–70).
  `stage2/metrics.jsonl`: eval 19레코드, eval_length 139–681 step(≈2.8–13.6s), **lap 완주 2회뿐**
  (step120k=19.3s, step140k=18.4s). **40–80s 완주대 = 0건.** 점프 패턴 확정.
- ❌ **"critic의 느린-완주 부재는 재학습으로 무력화"** (004:59). §2 참조 — stage2가 그 런 자체.
- ⚠️ **stage1 보조 증거**: 완주 13.3–27.9s, 509k step에 완주 159회, **60–80s 부재**(첫 완주 step162912=17.0s).
  warm-load 소스부터 "느린 완주"를 모른다.
- ✅ **warm-load 설계 자체**(004:56–57)는 정확히 구현됨 → 실패는 셋업 오류가 아니라 **시스템 동역학 본질**.
- ✅ **`stage1_map_easy3/latest.pt` 존재**(148MB).

### 향상 천장 / fast-and-surviving 재료 (D2 caveat)
- ✅ **004가 in-distribution 한계·caveat을 정직히 명시**(004:22–24, 53, 68–70). 인지 자체는 모범적.
- ❌ **"완주 6 + 부분진행으로 stitching 재료 충분"**: 265 ep 중 완주 **6(2.3%)**, 크래시 **259(97.7%)**.
  고속 step(>p75) 20,930개 중 **73.1%가 크래시로 종료**, 고잔여진행 동반 26.9%.
  **고속+고잔여진행 동시 충족 32-step 윈도 = 32 / 75,047 = 0.04%.**
  → value(=할인 return-to-go [sequence.py:138–147])는 "빠르면 곧 크래시→잔여 progress 낮음"을 학습 →
  **고속 구간을 회피**, 보수적 완주로 수렴. 천장이 데이터에 의해 낮게 고정.
- ⚠️ **완주 6개의 성격**: lap 17.3–18.0s지만 **평균속도 정규화 0.08–0.12로 보수적**(순간 max만 1.0).
  "빠른 완주"가 아니라 "조심해서 완주" → Diffuser가 빼낼 "어디서 빨라도 되는가" 정보가 빈약.

### D1 — 표현(centerline 피처 1순위)
- ✅ **Markov 근거**: Diffuser는 현재 1프레임만 조건([helpers.py:142–145] t=0 고정) → 피처가 raw lidar보다
  Markov-충족. 002 §2 권고와 정합.
- ✅ **pose 부재 확인**: npz 키에 `pose` 없음, `f1tenth_env.py:194–196`이 `reset(options={'pose'})` 지원
  → 재rollout로만 해소 → **피처 1순위면 P2(pose rollout)가 P3보다 선행 강제**(004:110 인지).

### D4 — value reward
- ✅ **progress(dense)+축소 collision(−2~−3), lap 보너스 제외, normed value[-1,1]** (004:77–79).
  002 §5 권고와 정확히 일치. 저위험.

### D5 / P0 — 글루 디커플
- ✅ **첫 블로커 = `rendering.py:4 import imageio`** 실측 확정. 체인:
  `temporal.py:7 → helpers.py:10(import diffuser.utils) → utils/__init__.py:6(from .rendering import *) → rendering`.
  006 ①이 1차 블로커를 정확히 지목.
- ⚠️ **006 목록 불완전 (숨은 체인 2건)**:
  1. `utils/__init__.py:6`이 9개 서브모듈을 **즉시 전부 import**, 그중 `colab.py:17 from .video import save_video`
     (→ `video.py:3 import skvideo.io`)는 **별도 import 체인**. rendering guard가 video까지 덮어야 함(006 미명시).
  2. **d4rl import 지점 3개**: `d4rl.py:25 import d4rl`(try/except L13–21 ✅) **앞**의
     `preprocessing.py:7 from .d4rl import load_environment`와 `sequence.py:7 from .d4rl import load_environment, sequence_dataset`
     은 의존순서상 먼저 평가됨 → ④ 스텁이 **함수까지** 커버해야 풀림(006 별도 명시 안 함).
- ✅ `buffer.py:12 np.int`(numpy1.24.4 에러), `buffer.py:78–79 timeouts assert`(termination_penalty=None로 침묵) — 정확.
- ❌ **"P0 게이트 = TemporalUnet import + 1-step loss로 충분"** (004:102): 1-step loss는 코어 import+forward만
  검증, **`train.py`의 `utils.Config/Trainer/batchify` 배선, `train_values.py`(ValueFunction), `plan_guided.py`(sampling)
  글루를 검증하지 못함**. 002 §11이 남긴 "scripts 글루 미감사"가 004에서 닫히지 않았다.
  추가 위험: `Config` pickle 덤프(`train.py:21,30,47,58,75`)→`config.py:17 importlib.import_module`이 직렬화 시점에
  Renderer 클래스를 끌어올 수 있음(NullRenderer 우회가 직렬화까지 안전한지 미확인).

### seam / normalizer / 게이트
- ✅ **action [-1,1] parity**: npz action min=−1.0/max=1.0(265 ep).
- ❌ **"normalizer 영속화 = Trainer.save pickle 덤프"** (004:86): `training.py:136–150` `Trainer.save`는
  **{step, model, ema}만 저장 — normalizer 미저장**. `serialization.py:36–60`은 **추론 시 dataset_config로
  normalizer를 재구성**(=데이터 재로드)한다. → 004 서술은 사실과 다르고, **추론(P5)은 동일 npz·동일 로더로
  normalizer를 재현**해야 한다(seam 리스크).
- ⚠️ **online 피처 parity 무보장**: P2(수집 시 피처) vs P5(추론 online 피처)가 **두 코드경로** →
  drift 구조적 위험. 004는 단일 피처함수 공유 장치를 명시하지 않음. (피처 계산 코드는 아직 양쪽 코드베이스
  어디에도 없음 → P3에서 신규 작성.)
- ⚠️ **clip_denoised 미언급**: `config/locomotion.py:119`가 설정 의존 → LimitsNormalizer([-1,1])면
  `config/f1tenth.py`에 `clip_denoised=True` 필요(002 §1.3).
- ⚠️ **K-step MPC**: P5(004:107)에 단어만 있고 주기·게이트·wall-clock 추정 없음.

---

## 4. 핵심 결함 Top 7 (심각도 + 완화책)

| # | 결함 | 심각도 | 완화책 |
|---|------|--------|--------|
| 1 | **D2 주경로(warm-load 재학습) 전제 반증** — 그 실험=stage2, 60–80s 완주 미산출. P1=알려진 실패 재현 | **Blocker** | **D2를 002 §3–4 결론으로 복귀**: 완주 가능한 빠른 policy에 speed-cap 스펙트럼{예 ×0.16/0.4/1.0} 적용해 **느린 완주 데이터를 *제작***. warm-load 유지 시 **P1 하드 게이트 + 실패 시 즉시 speed-cap 전환** |
| 2 | **fast-and-surviving 재료 거의 부재(0.04% 윈도)** → value가 느리고-안전 선호, 향상 천장 낮음 | **Critical** | speed-cap로 **풀-트랙 완주 커버리지**(완주 라인 유지) 확보 → 빠른 구간이 데이터에 실제 존재. P4 생성품질 점검으로 조기 판정 |
| 3 | **내부 모순: "expert 금지 + speed-cap 강등"인데 유일한 완주가 ~18s(=expert급)** → 폴백(D3)이 결국 expert-cap으로 붕괴 | **Major** | 프레이밍 정정: speed-cap은 폴백이 아니라 **"느린 완주 baseline 제작"의 주 도구**. 사용자 단서 "안 되면 그때만"의 *그때*는 metrics로 이미 도래 → P1 낭비 전 채택 |
| 4 | **baseline 미정의 게이트 없음** → P1이 DNF면 baseline 부재, ~18s면 expert급이라 Diffuser가 이기기 거의 불가 | **Major** | P1 게이트 = "목표대(예 ≥40s)에서 **안정적 완주** + lap 실측". 미달 시 speed-cap로 목표대 제작 |
| 5 | **normalizer 영속화 서술이 코드와 불일치**(Trainer.save가 normalizer 미저장) | **Major** | 학습 스크립트에서 normalizer **별도 pickle 저장** + 추론 로드, **또는** 추론이 동일 데이터로 재구성함을 명시·검증. P3 라운드트립 게이트 |
| 6 | **online 피처 parity 무보장**(수집 P2 vs 추론 P5 두 코드경로) | **Major** | **단일 피처함수 모듈**을 로더(P3)·online(P5)이 공유 import + golden-value 테스트(같은 입력→같은 피처) |
| 7 | **P0 게이트 불충분 + 디커플 목록 누락**(colab.py/preprocessing.py; train_values·plan_guided 미검증) | **Major** | P0 게이트 확장: (A) TemporalUnet import, (B) f1tenth 로더로 `SequenceDataset` 1개 적재, (C) `train.py` 1-step 실손실, (D) `train_values.py`·`plan_guided.py` import/파서 통과. (Config pickle→Renderer 끌림 확인) |

**추가(Minor~Major) — 표현 순서**: 피처 1순위는 pose 재수집을 강제(P2→P3 결합) + 곡률/자기교차 정확도
리스크를 키운다. **v1은 minimal-first(기존 265 ep + lidar 다운샘플~64, 재수집·pose·online피처 불요)로
P0→P6 파이프라인을 먼저 관통**시켜 *엔지니어링 리스크를 제거*하고, centerline 피처는 v2로 승격하는 게
개인 과제 규모에 합리적. **단 데이터(D2) 문제는 표현과 무관하게 별도 선결** — minimal-first는 코드 검증일
뿐, 97.7% 크래시 데이터면 결과는 여전히 나쁘다.

**Minor 누락 메우기**: `clip_denoised=True` / K-step(4~8) MPC 주기를 `config/f1tenth.py`·P5 스펙에 명시.

---

## 5. 누락 단계 / 게이트 (v3에 신설할 것)

1. **[P1 하드 게이트]** "behavior policy 제작" 성공 조건을 측정가능하게: *목표 lap-time대에서 N/M회 이상
   완주*. 미달 시 자동으로 speed-cap 경로 전환(폴백을 "수동 재고"가 아닌 **게이트 분기**로).
2. **[baseline 정의 게이트]** baseline = "P1이 확정한 *완주하는* 느린 policy의 lap time". DNF면 무정의 →
   진행 금지(speed-cap로 정의 가능 baseline 확보 후 진행).
3. **[P0 확장]** §4-#7의 (A)~(D). `Config` pickle 직렬화가 Renderer를 끌어오는지 P0에서 확인.
4. **[P3 normalizer 라운드트립]** 저장→로드→동일 bounds 검산 게이트(§4-#5).
5. **[P5 피처 parity 검산]** 수집-시 피처 == online 피처 golden test(§4-#6).
6. **[P5 K-step MPC]** 주기 K 명시 + wall-clock 추정(≈ lap step수 × n_diffusion(20) × n_guide(2) / K).
   sim-time이 lap time이고 wall-clock은 별도 보고임을 보고서에 명시.

---

## 6. ★ v3 계획 수정 지침 (다음 계획 에이전트용 diff 방향)

```
[D2 — 전면 재작성]  ★ Blocker
- "주=Dreamer warm-load 재학습 → mid-checkpoint를 느린 완주 behavior policy로.
-  critic의 '느린 완주 부재'는 재학습으로 무력화."
+ "[반증] warm-load 재학습 = 이미 실행된 stage2_oschersleben 그 자체(stage2_watchdog.sh).
+  곡선은 크래시(97.7%)→고속완주(~18s) 점프, 60–80s 완주대 부재. stage1도 동일.
+  → 느린 완주 baseline은 *제작*한다: 완주 가능한 policy(Dreamer가 내는 ~18s 완주 checkpoint,
+  필요 시 expert)에 speed-cap 스펙트럼{×0.16/0.4/1.0} 적용해 풀-트랙 완주 데이터 생성.
+  expert 금지 단서의 '안 되면 그때만'은 metrics로 이미 충족 → P1 낭비 전 speed-cap 채택.
+  (warm-load 고집 시: P1 하드 게이트 + 즉시 폴백 분기 명시.)"

[향상/천장 §]
- "재학습 policy 다양성으로 빠른 재료 확보"
+ "데이터에 고속+생존 재료가 0.04%뿐 → value가 느리고-안전 선호.
+  speed-cap 완주 스펙트럼으로 풀-트랙 커버리지 + 빠른 구간을 데이터에 실제 포함(투명 기술)."

[Phase / 게이트]
+ P1 게이트: '목표대(≥40s)에서 안정적 완주 + lap 실측'. 미달 시 speed-cap 자동 전환.
+ baseline 게이트: 완주하는 느린 policy의 lap. DNF면 진행 금지.
+ P0 게이트 확장: TemporalUnet import + f1tenth 로더로 SequenceDataset 적재
+   + train.py 1-step 실손실 + train_values.py/plan_guided.py import·파서 통과.
+   (Config pickle→importlib Renderer 끌림 P0에서 확인.)
+ P3: normalizer 저장/로드 라운드트립 검산(Trainer.save는 normalizer 미저장 — 별도 저장 필요).
+ P5: 단일 피처함수 공유(수집=추론) + golden test, K-step(4~8) MPC + wall-clock 추정.

[D1 — 순서]
+ "v1=minimal-first(기존 265ep + lidar 다운샘플~64, 재수집·pose 불요)로 파이프라인 관통.
+  centerline 피처는 v2 승격(pose 재rollout 후). 데이터(D2)는 표현과 독립적으로 선결."

[D5 / config]
+ config/f1tenth.py: clip_denoised=True(LimitsNormalizer[-1,1]), termination_penalty=None.
+ 006 디커플 목록에 colab.py:17(video 체인), preprocessing.py:7(d4rl 함수 import) 추가.
```

**재설계 시 핵심 의사결정 분기(에이전트가 사용자와 확정할 것)**:
- (A) **권장**: speed-cap-on-completing-policy를 *주경로*로. Dreamer가 내는 ~18s 완주 checkpoint를
  쓰면 "Dreamer-first" 사용자 의도도 충족(라벨상 'expert' 파일이 아니어도 속도는 동일대). 가장 빠르고 확실.
- (B) warm-load 유지하되 **reward-shaping 짧은 재학습**(속도 페널티+완주 보상)으로 느린 완주 policy를
  *공학적으로 제작* → speed-cap보다 공수 큼, 효과 불확실. 개인 과제 규모엔 과投.
- (C) 004 원안 그대로 P1 강행 → metrics가 예고한 실패 재현 위험. **비권장.**

---

## 7. 확인 못 한 항목 (침묵 금지)

1. **002 §4 수치 일부 불일치(정직 보고)**: 002는 "첫 완주 step129k, lap 37.6s, 완주 6회 전부 129k+"라 했으나,
   이번 검증에서 stage2 metrics의 eval 완주는 2회(120k=19.3s, 140k=18.4s), train_eps npz 완주 6개는
   **17.3–18.0s**다. **37.6s/129k는 재현되지 않음**(002 오기 또는 다른 필드/런 가능성). 단 *질적 결론
   (느린 완주대 부재, 완주는 모두 고속)*은 영향 없고 오히려 강화(완주가 ~18s라 더 expert급).
2. **stage2 런 완결 여부**: 마지막 eval ~step166k(=165994) → 500k 미도달(조기 종료) 추정. 단 stage1이
   509k 돌고도 60–80s 완주 부재 → 결론 불변.
3. **`log_lap_time_s=0.0` 의미**(미완주 vs 미측정) 미확정 → 완주 판정은 `log_completed`로 교차.
4. **새 재학습의 actor-critic 초기화 분포**(uniform/zero) — 곡선 형태 영향 가능하나 stage1/stage2 동일 패턴.
5. **state[3] 정규화→물리 m/s 역산 계수** 미확정(정규화 [-1,1]을 프록시 사용). "얼마나 더 빨라야" 정량화 영향.
6. **NullRenderer 스텁이 Config 초기화 부수효과 없이 mujoco 체인을 우회하는지** 실런 미검증(정적 분석만).
7. **1-step loss 수치적 안정 수렴** 미실행(import 체인만 실측).
8. **online closest_idx 자기교차 점프 방지**(`f1tenth_env.py:261` 윈도 추적) 정확도 — 테스트 하니스 미검토.

---

## 8. 다음 단계 (인수인계)

1. **이 문서 = D2 재설계의 입력.** 다음 계획 에이전트는 §6 분기 (A)/(B)/(C) 중 사용자와 확정 후 v3 작성.
2. **P0는 D2와 무관하게 즉시 선행 가능** — 단 게이트를 §5-3으로 확장(006 디커플 + colab/preprocessing 추가).
3. v1은 minimal-first(lidar 다운샘플 + 기존 265 ep)로 P0→P6 관통 권장, 피처는 v2.
4. 분기마다 _thinking 문서 + commit + push (상시 규약). f1tenth 판단 시 [[003-project-spec]] + PDF 참조.

# 006 — Diffuser offline RL 확정 계획 v3 (데이터 = Dreamer speed-cap 완주 스펙트럼)

> 2026-06-18. plan_new/004(v2)를 잇는 **현재 SSOT**. 005 적대적 critique를 전면 반영하고, 사용자와
> 합의한 데이터 설계(speed-cap 완주 스펙트럼 + 소량 expert)를 확정한다. **004의 D2는 본 문서 D2로
> 폐기·대체.** 나머지 D1/D3/D4/D5는 004 + critique 보강. **모든 lap time은 2랩 기준(사용자 규약).**
> **expert(고속 tier) 포함은 교수님 허용 가정**(미확정 — fallback 분기 명시). append-only.
> 선행: [[004-diffuser-plan-v2]], [[005-diffuser-critique-v2]], [[006-glue-correction-and-data-contract]],
> [[005-diffuser-venv-and-handoff]], [[004-dreamer-reuse-and-behavior-policy]], [[003-project-spec]] + PDF.

---

## 배경 (이 문서만 읽어도 되도록)

**과제**(AIE4003 개인, 주제1 Offline RL): 느린 policy로 데이터 수집 → **환경 추가 상호작용 없이** 더
빠른 주행 정책 학습. 산출물 = behavior policy보다 빠른 2랩 lap time + 보고서. 트랙 = Oschersleben 단일.

**모델**: Diffuser(Janner ICML2022, clone `~/planning_with_diffusion`). 궤적 diffusion + 별도 value
함수 → 현재 관측 조건으로 미래 궤적 생성, value gradient가 고-return으로 유도(MPC, 첫 action 실행).
코어 무변경. **향상 원리·한계**: stitching + value guidance가 데이터 내 좋은 구간을 재조합해 baseline을
넘는다. 단 **in-distribution 생성기라 데이터에 없는 속도/전이는 외삽 불가** → 데이터 설계가 성패를 결정.

**평가**: 정성적(교수님). "60→30(2랩) 가능", medium-expert/medium-replay식 혼합 허용.

---

## 004 → 006 변경 요지 (delta)

| # | 변경 | 근거 |
|---|------|------|
| Δ1 | **D2 = "Dreamer warm-load 재학습"(폐기) → "best 완주 정책을 speed-cap한 완주 스펙트럼"** | 005 §2: warm-load 재학습 = 이미 돌아간 stage2(크래시97.7%→고속완주~18s/랩 점프, 느린완주 부재) |
| Δ2 | **데이터 = 완주 데이터만**(크래시 데이터 배제) | value=return-to-go라 "빠르지만 크래시"는 회피됨 + in-dist라 "고속 생존" 외삽 불가 (대화 검증) |
| Δ3 | **소량 expert tier 포함(45/45/10, 잠정)** = medium-expert식 구성 | offline RL 표준 세팅(D4RL). 천장 확보 + 개선의 정직성 |
| Δ4 | P0 게이트 확장·디커플 목록 완성·normalizer 서술 정정·online 피처 parity·baseline 게이트 | 005 §4–6 |
| Δ5 | 모든 lap time 2랩 표기 | 사용자 규약(2026-06-14) |

---

## 핵심 검증 사실 (2랩 기준)

- **lap 단위**: npz `log_lap_time_s`=랩당, `에피소드 steps × 0.02`=2랩 총합. 완주(log_completed=1)=2랩.
- **Dreamer 완주 실측**(replay 완주 6개): **2랩 ≈ 35.5–37.6s**(랩당 17.3–18.8s). 모두 "조심해서 완주"(순간 max만 1.0).
- **★ 체크포인트 인벤토리**(`runs/stage2_oschersleben`, 실측): 저장된 **완주 정책 = `policy_best_lap16.6s_step85k`(랩당16.6s≈2랩~33s)·`...lap17.4s_step80k`(≈2랩~35s)뿐.** 인터벌 스냅샷 step_5k~85k는 002대로 **대부분 크래시**(완주 안 함). → **"느리게 완주하는 정책"은 존재하지 않는다.** 사용자가 기억한 "38s 정책"은 로드 가능한 체크포인트가 아니라 **replay 에피소드(37.6s/2랩)**다.
- **함의**: 모든 tier(느린~빠른)는 **동일한 best 완주 정책을 speed-cap한 것**으로 만든다(단일 소스 = 분포 일관). "38s expert tier"는 best 정책을 *가볍게 캡*해 얻는 **목표 2랩 시간**이지 별도 정책이 아니다.

---

## 설계 결정

### D1. 관측 표현 — minimal-first(lidar 다운샘플), pose는 기록만 *(005 §4 권고)*
- **v1 = lidar 다운샘플(~64) + state.** 기존 파이프라인을 P0→P6로 *관통*시켜 엔지니어링 리스크(피처 online 계산·곡률·자기교차 인덱스) 제거. raw lidar+state 그대로 저장.
- **단 speed-cap rollout(P2) 때 pose+raw속도+trackname도 함께 기록**(어차피 새로 도니 공짜) → centerline 피처(~10D)는 **v2 승격 옵션**으로 열어둠. 표현 교체 시 재수집 불요.
- 근거: Diffuser는 현재 1프레임만 조건([helpers.py:142–145])이라 Markov 표현 선호 — 피처가 이상적이나 lidar로도 v1 충분.

### D2. ★ 데이터소스 — best 완주 정책의 speed-cap 완주 스펙트럼 *(004 D2 전면 대체)*
**반증 요약**(005 §2): "warm-load 재학습으로 느린 완주 policy 제작"은 불가 — 그 실험이 곧 stage2이고
크래시→고속완주로 점프, 느린 완주대(2랩 40–80s) 0건. 저장 스냅샷도 느린-완주 없음.

**확정 설계**:
- **소스 = `policy_best_lap16.6s_step85k`** (유일하게 안정 완주하는 정책, 2랩 ~33s). 단일 소스 → 라인·동역학
  일관된 분포(GapFollower 등 타 소스 혼합 배제 — 이중 라인 stitching 위험·서사 일관성).
- **방법 = raw-stage speed clamp** (D3): 정책 출력을 raw로 역정규화 → **raw speed를 ≤ X m/s로 상한
  클램프**(κ 곱 아님 — 코너 감속은 두고 최고속만 자름) → 재정규화. steer 무수정. env·차체물리 무변경.
- **tier 구성(잠정 비율, 상황 따라 조정)**:

  | tier | 목표 2랩 | 캡(잠정, P1 보정) | 비율 | 역할 |
  |---|---|---|---|---|
  | 느림(baseline) | 최장(P1 실측, 예 ~60s+) | ~10 m/s | 45% | behavior policy / 이길 대상 |
  | 중간 | 중간(P1 실측) | ~15 m/s | 45% | 커버리지 |
  | 빠름(expert tier) | **~38s** | best 정책 가벼운 캡(~18 m/s) | 10% | fast-and-surviving 재료 / 천장 |

- **"38s expert" 정직성 선택(사용자 확정)**: best 정책의 *raw 무캡(~33s)* 대신 **~38s로 살짝 캡한 것을
  최고 tier로** 둔다. 이유: 더 약한 expert를 쓰면 "최고 정책을 베껴서"가 아니라 **offline RL로 정당하게
  끌어올렸다**는 서사가 깨끗해진다(개선이 더 *정직하게* 드러남). ⚠️ 정정: 이는 절대 개선폭을 키우는 게
  아니라 천장을 ~33s→~38s로 낮추는 선택 — 효과는 "정직성"이지 "더 큰 개선"이 아님(사용자 인지·수용).
- **expert tier는 직선·코너 *모두*에서 "빠르고 생존"한 재료** → 인위적 천장(중간 캡에 갇힘) 제거, value
  guidance의 고-return 타겟 제공. 소량(10%)이라 Diffuser가 복제가 아니라 stitch/guide로 끌어올려야 함.
- **교수님 허용 가정**. **불허 시 fallback**: expert tier(10%) 제거 → {느림·중간}만(천장 = 중간 캡 속도,
  개선 작지만 진짜). 비율·캡은 P4 결과 보고 조정.

### D3. speed-cap — 주 도구로 승격 *(004는 fallback이었음)*
raw-stage 상한 클램프(위). κ∈고정값 아님, **각 tier의 실제 2랩 시간·완주 여부는 P1 rollout 실측**으로
확정. ⚠️ 리스크: 고속(20m/s)용으로 학습된 조향을 저속에 쓰면 오버스티어 등 미스매치로 완주 실패 가능
→ 캡별 완주 실측이 게이트.

### D4. value 대상 reward *(004 유지, 005 §3 ✅)*
progress(dense, `log_reward_progress`) + 축소 collision penalty(−2~−3) + **lap 보너스 제외** +
**normed value([-1,1], ValueDataset normed=True)**.

### D5. Diffuser 글루 *(006-glue + 005 보강)*
- **P0 디커플 목록 = [[006-glue-correction-and-data-contract]] §2 + 005 §3-D5/P0 추가분**:
  ① `utils/rendering.py` heavy import(imageio:4/mujoco_py:8/`from .video`:13/d4rl:15) guard
  ② **`colab.py:17 from .video`(skvideo 체인)도 guard** ③ `d4rl.py:25 import d4rl` try/except +
  **`preprocessing.py:7`·`sequence.py:7`의 d4rl *함수* import까지 스텁 커버** ④ `buffer.py:12 np.int→np.int64`
  ⑤ `sequence.py:22–23 load_environment/seed` 스텁 + `datasets/f1tenth.py` 로더 + `config/f1tenth.py`.
- **normalizer 영속화 정정(004 오류)**: `Trainer.save`는 {step,model,ema}만 저장, normalizer 미저장
  ([training.py:136–150]); 추론은 dataset_config로 **재구성**([serialization.py:36–60]). → 학습 시
  normalizer **별도 pickle 저장** + 추론 로드, **또는** 추론이 동일 데이터·로더로 재구성함을 보장.
  **P3 라운드트립 검산 게이트.**
- **online 피처 parity**(피처 표현 채택 시): 수집(P2)·추론(P5) **단일 피처함수 모듈 공유** + golden test.
- config: `clip_denoised=True`(LimitsNormalizer [-1,1]), `termination_penalty=None`(또는 `timeouts=is_last&~is_terminal` 합성).
- 평가 루프 `scripts/plan_f1tenth.py` 신설, device 파라미터화. 코어 무변경.

---

## 성능 기대치 / 성공 기준 (정직하게 — floor / realistic / stretch)

| 수준 | 내용 | 달성성 |
|---|---|---|
| **Floor(성공 기준)** | 느린 baseline tier(~60s+)보다 빠른 2랩 완주 | **달성 가능** — 과제 "behavior policy 대비 개선" 충족 |
| **Realistic(강한 결과)** | 데이터 90%가 훨씬 느린데도 Diffuser가 **expert tier(~38s) 수준에 근접** | offline RL stitching/guidance의 정수 — **보고서로 당당한 결과** |
| **Stretch(사용자 기대)** | **expert를 *뛰어넘기*(<38s, 나아가 <33s)** | ⚠️ **구조적으로 어려움** |

⚠️ **"expert 뛰어넘기"에 대한 정직한 평가**: 모든 tier가 **동일 정책을 균일 스케일한 것**이라, stitching이
"어느 단일 궤적보다 빠른" 결과를 내려면 *서로 다른 구간에서 빠른* 상보적 궤적이 필요한데 이 데이터엔 없다
(최고 tier가 모든 구간에서 가장 빠름). Diffuser는 in-distribution이라 **데이터 최속을 넘기 어렵다** →
현실적 최선 ≈ expert tier 근접. **expert 초과를 진짜 원하면 *라인/구간 다양성*(서로 다른 정책·라인)이
필요** = v2 과제. v1의 당당한 서사 = **"대부분 느린 데이터에서 expert급 주행을 복원·합성"**(이게 이미 강한 주장).
사용자 합의: 기대는 하되 "안 되면 그때 재고".

---

## 데이터-모델 궁합 (★ 보고서 필수 서술 — 사용자 지정)

> 우리 데이터는 "매우 느림(저속 캡) + 빠름(expert tier)"의 **이중분포(bimodal/medium-expert)**다. 단순
> behavior cloning은 두 모드를 평균 내 망가진다. 그러나 **value 기반·궤적 생성 방법(Diffuser 포함)은
> 이런 혼합 데이터를 다루도록 설계**돼 있어, value guidance로 고-return 모드를 선택·stitching으로 좋은
> 구간을 재조합한다. 다만 unimodal 데이터보다 일반적으로 *더 어려운* 세팅이며, **그래서 이 과제에
> Diffuser 선택이 적절하다.** (D4RL의 medium-expert/medium-replay가 바로 이 세팅의 표준 벤치마크.)

---

## Phase 분해 (게이트 + 런타임 배치)

| # | 작업 | venv | 게이트 |
|---|------|------|--------|
| **P0** | 글루 디커플(D5) + 더미 로더로 `train.py` 1-step 실손실 | **새 .venv** | (A) `from diffuser.models.temporal import TemporalUnet` 통과 (B) f1tenth 로더로 `SequenceDataset` 1개 적재 (C) train.py 1-step loss (D) `train_values.py`/`plan_guided.py` import·파서 통과. Config pickle→Renderer 끌림 확인 |
| **P1** | best 정책 speed-cap **캡↔2랩시간 캘리브레이션** + 각 tier **완주 실측** | **RL_project venv** | 목표대(느림/중간/~38s)에서 **안정적 완주** + 2랩 실측. **baseline = 느린 tier(cap~10)의 2랩시간으로 정의**. 캡이 완주 실패 시 캡 조정 |
| **P2** | 확정 캡들 rollout, **lidar+raw+pose+trackname 기록**, 45/45/10 수집(크래시 에피소드 폐기) | **RL_project venv** | tier별 분포·완주율 리포트 |
| **P3** | `datasets/f1tenth.py` 로더 + lidar 다운샘플 + **normalizer 저장/로드 라운드트립 검산** | 새 .venv | SequenceDataset 통과 + 라운드트립 동일 bounds |
| **P4** | 궤적 diffusion + value 학습, **생성 궤적 품질 점검**(expert tier 활용·빠른구간 합성 여부) | 새 .venv | loss 수렴 + 생성 품질. expert 비율 부족 시 조정 |
| **P5** | `plan_f1tenth.py`: GuidedPolicy↔f110_gym, normalizer 로드, **K-step(4~8) MPC** + wall-clock 추정 | **새 .venv** | 1 ep 완주 시도 |
| **P6** | 평가: baseline(느린 tier) 대비 2랩 lap time, 2랩 완주, expert tier 근접 여부 | 새 .venv | **baseline보다 빠름(floor)**; expert 근접(realistic) |

- **P0는 데이터·교수님 무관하게 즉시 선행 가능.** 데이터생성(P1/P2)=RL_project venv(Dreamer 본거지), 학습·평가=새 .venv.

---

## 리스크 Top + 완화
1. **저속 캡 완주 실패**(조향-속도 미스매치) — P1 캡별 완주 실측 게이트, 실패 시 캡 조정.
2. **expert 10%가 신호로 부족** → P4 생성 품질 점검 후 비율 상향. (비율 잠정.)
3. **expert 초과 불가**(균일 스케일 한계) → v1 목표를 "expert 근접/baseline 대비 개선"으로, 초과는 v2(라인 다양성).
4. **글루 디커플 잔여**(rendering/colab/preprocessing 체인) → P0 확장 게이트로 일괄.
5. **normalizer/피처 seam**(학습↔추론) → normalizer 라운드트립(P3), 피처 단일함수 공유+golden test(P5).
6. **공유 gym 소스(`RL_project/gym`) 수정 금지** — RL_project에 영향.

---

## 다음 단계 (인수인계)
1. **즉시 = P0**(새 .venv): 006-glue §2 + 위 D5 디커플 → 코어 import + train.py 1-step loss + train_values/plan_guided 파서.
2. **교수님 확인 1건**: expert(고속 tier, ~38s) 정성평가 포함 가능 여부 → 불허 시 D2 fallback(expert tier 제거).
3. P1: best 정책 speed-cap 캘리브레이션 + baseline 2랩 실측(RL_project venv).
4. 분기마다 _thinking 문서 + commit + push(상시 규약, 자율 pull 금지). f1tenth 판단 시 [[003-project-spec]] + PDF.

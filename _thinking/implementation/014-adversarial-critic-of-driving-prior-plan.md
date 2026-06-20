# 014 — 적대적 검수: "주행 prior + 충돌 value" 역할분리 계획 (013 대상)

> 2026-06-20. 검수 대상 = [[013-p6-failure-diagnosis-and-driving-prior-plan]] §3 역할분리 계획 + §4 데이터효율.
> 임무 = 계획을 *통과*시키는 게 아니라 *깨뜨리는* 것. 모든 판단은 npz·코드 1차 소스 직접 확인.
> **다음 세션 인수인계용.** append-only. (검수자: critic 세션, 코드/실험 무변경, 비파괴 조회만.)

---

## 0. 한 줄 판정

**판정 = (B) 조건부.** 계획의 핵심 직관(crash 궤적을 prior에서 빼면 P6의 degenerate 후진/스핀이
사라진다)은 **1차 소스로 타당**하다. 그러나 계획이 *진짜 개선*(<56s)의 엔진으로 지목한 **value
guidance는 보상구조상 "충돌로 가는 고속"을 "안전 고속"보다 4배 선호**(D3, 아래 정량 확증)하므로
**역할분리만으로는 진짜 개선이 안 나온다.** 또 prior에 넣을 고속 재료(249개)는 **충돌 직전 공격
주행**이라 닫힌 루프에서 lap-2 충돌을 재현할 위험이 크고, **데이터 믹스가 고속-충돌 lap1로 73%
편중**돼 "안전 56s" 층위조차 자동 보장되지 않는다. → **3개 조건 충족 시에만 현실적**(§9).

---

## 1. 검증 방법 (1차 소스 목록)

- 데이터: `runs/crash_data/{cap5,cap5_full,cap10,cap10_full,cap15,cap20}/*.npz` 직접 적재·집계
  (스크립트 `/tmp/critic_data_audit.py`, `/tmp/critic_lap2_d3.py` — RL_project `.venv` numpy1.24).
- 로더: `vendor/diffuser/diffuser/datasets/f1tenth.py` (전문 읽음).
- config: `vendor/diffuser/config/f1tenth.py` (discount 0.99, normed=True, H128, SafeLimitsNormalizer).
- value guidance: `diffuser/sampling/guides.py`, `diffuser/utils/serialization.py:62` check_compatibility,
  `diffuser/datasets/normalization.py:152-191` Limits/SafeLimitsNormalizer.
- 평가: `scripts/plan_f1tenth.py` (K=1 MPC = `policy(cond)` 매 step, action_raw 1개 사용).
- 진단 스크립트(잔존): `/tmp/p5_gen_diag.py`·`p5_cl_diag.py`·`p5_k_test.py`·`p5_value_test.py` (전부 읽음).

---

## 2. ★ 데이터 주장 검증 — 249개는 **사실** (linchpin 통과)

`log_lap_time_s>0` 첫 시점까지 truncate한 결과를 전 tier 직접 집계(`critic_data_audit.py`):

| tier | nfile | crash | 완주(log_completed) | lap1>0 ep | lap1 시간(median) | lap1 길이(median) |
|---|---|---|---|---|---|---|
| cap5 | 13 | 13 | 0 | 5 | 58.1s | 2908 |
| cap5_full | 31 | 9 | 22 | 25 | 57.9s | 2896 |
| cap10 | 21 | 21 | 0 | 14 | 28.7s | 1434 |
| cap10_full | 40 | 10 | 30 | 37 | 28.6s | 1432 |
| **cap15** | 371 | 371 | 0 | **132** | **19.7s** | 988 |
| **cap20** | 291 | 291 | 0 | **117** | **18.0s** | 903 |

- **cap15 132 + cap20 117 = 249** 고속 lap1, 시간 19.7/18.0s → **013 §4.1 표와 정확히 일치. 확증.**
- 완주(log_completed) = **52** (cap5 22 + cap10 30) → 013 일치. 확증.
- **추가 발견(013 누락)**: cap5/cap10 충돌 ep에도 lap1이 있어 **전체 lap1>0 = 330개**. 즉 prior 후보는
  완주 52 + truncate lap1 278(cap5 5+cap5f 3+cap10 14+cap10f 7+cap15 132+cap20 117) ≈ **330 궤적**.
- truncate가 "충돌 前"인지: log_lap_time_s>0은 *완주 순간 기록*이므로 그 직전까지 자르면 정의상 충돌
  이전 구간. (b) 18~20s 고속도 확증. → §3에서 (c) "충돌 직전 공격 패턴 혼입" 여부를 본다.

**소결**: 013의 "재수집 0으로 고속랩 249개" 숫자 주장은 **거짓 없음.** linchpin 데이터는 실재한다.

---

## 3. ★ 의문4(추출 함정) — 고속 lap1은 "지속가능한 빠름"이 아니라 "충돌-직전 공격 주행"

`critic_lap2_d3.py`로 *각 고속 ep가 lap1 완주 후 lap2 어디서 충돌하는지* 측정:

| tier | lap2 즉시충돌(<10% 진행) | lap2 절반이상 진행 | lap2 진행률 median |
|---|---|---|---|
| cap15 | **27%** | 14% | 0.29 |
| cap20 | 11% | **52%** | 0.51 |

- cap15 고속랩의 **27%가 lap2 시작 즉시(10% 미만) 충돌**, 절반 이상 버틴 건 14%뿐. → cap15 19.7s
  페이스는 **마진이 거의 없는 한계주행**. cap20(18.0s)이 오히려 더 지속가능(52%가 lap2 절반↑).
- **속도 프로파일(state vx, _STATE_SCALE=20 정규화값)**: lap1 내에서 *후반·마지막 50step이 가장 빠름*
  (cap20 전반 0.68→후반 0.75→마지막50 0.79 ≈ 13.6→15.0→15.8 m/s). 즉 **lap1 말미에 최고속으로
  가속한 뒤 lap2에서 충돌** = 추출 구간의 꼬리가 가장 공격적. → 의문4의 (c) "충돌 직전 공격 패턴 혼입"
  **사실로 확인.** 다만 상승폭은 16%로 폭발적 스파이크는 아님(치명적까진 아니나 분포 오염은 실재).

**판단**: lap1은 truncate로 *기하학적으론* 깨끗(충돌 프레임 없음)하나, **주행 스타일이 "한계에서
1랩만 버틴" 정책의 것**이다. 이 스타일을 prior가 모방하면 닫힌 루프가 그 페이스로 달리다 **lap2에서
원본과 같은 충돌을 재현**할 개연성이 높다. (불확실성: 원본 충돌은 stochastic rollout 노이즈가 일부
원인 → deterministic 평가가 더 나을 여지 있음. 그러나 닫힌 루프 diffusion도 per-step std~0.02 +
covariate shift가 있어 마진 부족은 그대로 위협.) **= 이것이 "stretch <56s"의 1순위 리스크.**

---

## 4. ★★ 의문2(D3) — 역할분리는 D3를 **풀지 않는다. 옮길 뿐이다** (정량 확증)

013 §2는 "D3 아님(scale 0~1 다 충돌)"이라 했지만 이는 *오독*이다. P6에서 scale=0도 충돌한 건
**prior 자체가 crash-prone**이라서지 D3가 없어서가 아니다. D3(=value가 고속-충돌과 고속-안전을
구분 못함)는 **value의 보상구조 문제**이고, 역할분리는 prior만 청소할 뿐 value는 그대로 둔다.

**할인 return 직접 계산**(γ=0.99, H=128, `critic_lap2_d3.py`):

| 윈도 종류 | n | 할인 return mean | min | max |
|---|---|---|---|---|
| **고속-충돌**(cap20, 충돌 직전 128step) | 80 | **32.86** | 16.6 | 112.1 |
| **저속-안전**(cap5 완주 윈도) | 633 | **8.12** | 3.8 | 53.4 |

- 분해: 128step **할인 progress 합 = +23.25** ≫ **할인 충돌 penalty = −10·γ¹²⁸ = −2.76.**
- → **value는 "충돌로 끝나는 고속 윈도"를 "안전 저속 완주"보다 return 4배 높게 평가한다.** progress
  보상(~0.3/step)이 128step 누적되면 +23인데, 충돌 −10은 γ=0.99로 128step 할인되면 −2.76로 소멸.
  **value guidance는 prior를 "더 빠른=더 충돌 쪽"으로 민다.** normed=True는 단조변환이라 이 순서 보존.
- 현 value(corr 0.98)는 *이 오염된 return에 0.98로 잘 맞춘* 것 = **충실하게 틀린** 모델. 013 §6의
  "value 재사용해도 됨(corr 0.98)"은 **D3를 그대로 들여오는 것**이라 위험.

**판단**: 역할분리 계획의 §3 "value가 충돌 영역 *밖에서* 속도 유도" 전제는 **데이터로 반증된다.**
보상구조상 충돌 영역은 value에게 *고-return 영역*이다. **prior가 깨끗해서 value가 prior를 실제
충돌 궤적으로 밀진 못해도, prior의 *가장 빠른 모드*(=cap15/20 lap1, §3에서 본 한계주행)로 밀어
그게 닫힌 루프에서 충돌한다.** 즉 "scale 키우면 충돌"은 P6에서 끝이 아니라 **새 prior에서도 재현될
구조적 문제.** value guidance를 진짜 개선 엔진으로 쓰려면 **D3를 먼저 고쳐야 한다**(§9-2).

---

## 5. 의문3(진단의 유일성) — "prior 오염"은 진짜 원인이나 **유일하지 않다**. K=1 구조문제 잔존

013 §2는 근본원인을 "closed-loop compounding error + crash-poisoned prior"로 *둘 다* 적었으나,
계획 §3은 **prior 청소 한 축만** 처방한다. 1차 소스로 본 두 메커니즘:

- **(a) crash-poisoned prior (청소로 해결됨)**: 데이터 93%가 짧은 충돌이라 prior의 "전형적
  continuation = 멈춤/후진." 이게 P6의 mean cmd −4.0(후진) degenerate의 직접 원인. → **깨끗한
  prior면 이 후진 degenerate는 사라질 것**(타당, 계획의 진짜 강점). 단 이건 *안전 층위*를 살릴 뿐.
- **(b) mixture-averaging + K=1 compounding (미해결)**: 출발 obs(rest)는 cap5~cap20 전 tier가
  *동일*하므로 conditioning이 속도를 구분 못함 → diffusion은 전 tier continuation의 *평균*을 생성.
  새 prior 믹스는 **고속(249)이 73% 지배**(§6)라 평균이 고속 쪽으로 쏠림 → 출발부터 한계페이스로
  가속 → §3의 lap2 충돌 위험으로 직행. K=1 재계획의 compounding(작은 불일치 누적)은 prior 품질과
  *독립*인 구조문제로, gen_diag가 "정확 obs엔 정확 생성"인데도 닫힌 루프가 발산한 게 그 증거.
  계획은 "K·scale 재튜닝"만 적고 **구조적 처방(warm-start 등)이 없다.** K-step test(`p5_k_test.py`)가
  보인 트레이드오프(K=1 degenerate ↔ K≥10 open-loop-blind 84step 충돌)는 깨끗한 prior로 *완화*될 수
  있으나 *제거*된다는 증거는 없다.

**판단**: 진단은 (a)를 정확히 잡았고 청소가 (a)를 고친다. 그러나 (b)는 미배제·미처방. **깨끗한
prior가 (b)까지 자동으로 고친다는 보장이 1차 소스에 없다.** → warm-start를 권고(§9-3).

---

## 6. ★ 새 발견(시드 외) — 데이터 믹스가 고속-충돌 lap1로 **73% 편중** → "안전 56s"도 자동 아님

013 §5는 "안전 층위 = 주행 prior(완주 *중심*) → ~56s 모방"이라 하나, 실제 prior 믹스를 집계하면:

- 완주 52 中 **56s대(cap10) = 30개**, 115s대(cap5) = 22개. → "baseline 초과+모방"의 핵심 재료
  (cap10 완주)는 **30개뿐.**
- 고속 lap1 = **249개.** → prior의 **73%(249/~340)가 한계-고속 lap1**(§3에서 본 충돌-직전 스타일).

즉 prior는 "완주 중심"이 아니라 **"한계-고속 lap1 지배"**다. BC(scale=0)조차 이 믹스 위에선
고속 쪽으로 쏠려 §3의 lap2 충돌 위험을 안는다. **"안전 56s 모방"을 실제로 얻으려면 완주(특히
cap10)에 가중치를 주거나, 완주-only prior를 먼저 세워 닫힌 루프 안정성을 분리 검증**해야 한다.
"300개 다 넣으면 6배라 좋다"(§4.1)는 *episode 수*와 *닫힌 루프 robustness*를 혼동한 것.

---

## 7. 의문6(normalizer) — 계획의 진단·처방 **정확** (이 부분은 통과). "희석" 우려는 기우

1차 소스 확인:
- `serialization.py:62-73` `check_compatibility`는 `type(normalizers[key])`만 비교 → **mins/maxs
  stats 미검사**. driving-only stats와 full stats가 둘 다 SafeLimitsNormalizer면 **조용히 통과.**
  → 013 §6 주장 **정확.**
- `guides.py:12-14` `ValueGuide.forward(x,cond,t)` = `self.model(x,...)` 직접 호출 → diffusion이
  생성한 x(diffusion normalizer 공간)를 value가 *재정규화 없이* 자기 공간으로 가정. 불일치 시
  gradient 손상 → **prior/value normalizer 통일 필수**(013 처방 정확).
- "전체 stats로 통일하면 주행-only 학습 이점 희석" 우려(임무 의문6): **기우.** LimitsNormalizer는
  per-dim [min,max]→[-1,1] *선형 affine*(normalization.py:157-162). full stats로 정규화해도 prior가
  *학습하는 분포*(주행-only)는 불변, 좌표계만 바뀜. driving 데이터가 [-1,1]의 부분구간을 차지할 뿐
  손실 없음. → **통일 처방 채택해도 안전.** (단 value 재사용 시 §4의 D3가 따라옴은 별개 문제.)

**판단**: Q6는 계획이 옳다. 유일 보강 = "value 재사용=현 normalizer+현 D3 동반"임을 명시할 것.

---

## 8. 의문5(데이터 효율) — 양은 충분, 문제는 *커버리지*

- prior 후보 transition 수(median 길이×ep 추정): 완주 52 ≈ 214k(cap10 30×2864 + cap5 22×5800) +
  고속 lap1 249 ≈ 236k(cap15 132×988 + cap20 117×903) ≈ **~450k transition.** Diffuser 원논문
  도메인(D4RL ~1M, maze2d 등)보다 작지만 *희박하지 않다.* → **양(count)은 병목 아님.**
- 진짜 병목 = **닫힌 루프 분포 커버리지**(covariate shift). 이건 transition 수가 아니라 *닫힌 루프가
  방문하는 상태 주변의 데이터 밀도* 문제 → in-dist 데이터를 더 쌓아도 완전 해결 안 됨. 249 추가는
  *속도 다양성*은 늘리나(좋음) *robustness 커버리지*를 늘린다는 보장은 없다.
- 013의 "52→300 6배"는 ep 수 프레이밍이고, 그 6배의 대부분(249)이 *한계-고속*이라 robustness
  기여는 불확실. → 의문5 답: **count는 OK, 계획이 기대는 "6배니까 shift 메움" 인과는 약함.**

---

## 9. 계획을 살리는 최소 수정안 (우선순위 — over-plan 금지, 진짜 필요한 것만)

판정 (B)의 "조건"이자 처방. 1·2를 안 하면 진짜 개선은 거의 안 나온다.

1. **[필수·최저비용] 완주-우선 단계화로 "안전 층위"를 먼저 분리 검증.**
   - 1차 prior = **완주 52 (+선택적 cap10 완주 가중)** 만으로 학습 → scale=0(BC) 닫힌 루프 평가.
     목적: P6의 후진 degenerate가 *깨끗한 prior로 실제 사라지는지*, 56s대 완주가 나오는지를 **고속
     재료 변수 없이** 확정. 여기서 완주가 나오면 "baseline 초과"(보고서 문자적 성공)는 확보.
   - 그 다음에만 249 고속 lap1을 추가(stretch 실험). 한 번에 다 넣으면 §6 편중으로 원인분리 불가.
   - 비용 거의 0(데이터 이미 있음, 로더 플래그만). **닫힌 루프 안정성 ≠ 데이터 양임을 분리 입증.**

2. **[stretch의 전제·필수] value를 쓰려면 D3를 먼저 고쳐라.** 현 value(corr 0.98)는 고속-충돌을
   선호(§4). 후보(저비용 순):
   - (a) **충돌 penalty 증폭**: −10 → −100~−300. 그러면 −100·γ¹²⁸=−27.7 > +23 progress → 충돌
     윈도 return이 음수로 뒤집힘. **재구성은 npz 로그성분 가중합으로 가능**(reward 재조립), diffusion
     재학습 불요, value만 ~재학습. → 재계산으로 "고속-충돌 < 저속-안전" 뒤집혔는지 *검증 후* 사용.
   - (b) **value horizon 단축**(progress 누적↓ 상대적 충돌 부각) 또는 **할인 강화**(γ↓). 단 장기
     credit과 트레이드오프.
   - **이걸 안 하면 value guidance는 중립~유해**(scale↑=충돌). 그 경우 "진짜 개선"은 포기하고
     BC+속도다양성(고속 lap1로 prior가 자연히 빠른 모드 보유)에만 기대야 함 → stretch 확률 급락.

3. **[(b) compounding 대응] warm-start 샘플링** 도입(K·scale 재튜닝만으론 부족). 매 step diffusion을
   백지에서 뽑지 말고 *직전 plan을 조건/초기값*으로 이어 샘플 → K=1 degenerate와 K-large blind 사이
   sweet spot 확장. 코어 무변경 범위에서 sampling 글루로 가능(plan_f1tenth.py 레벨).

4. **[정합·이미 계획됨] normalizer 통일**(§7). prior/value 같은 normalizer 객체 공유. value 재사용
   시 그 value가 D3를 동반함을 명시(2와 충돌하면 2 우선 = D3 고친 value로 새로).

5. **[보고서 헤지] 고속 lap1 추가는 stretch 실험으로만, 안전 prior와 분리 로깅.** 어느 쪽이 닫힌
   루프를 깨는지(완주 vs 고속) 분리돼야 정성평가 서사가 산다.

---

## 10. 결론 — "이대로 실행하면 무엇이 가장 그럴듯한가"

**가장 그럴듯한 결과(현 계획 그대로, 300개 일괄 + value 재사용):**
- 깨끗한 prior로 **P6의 후진/스핀 degenerate는 사라진다**(§5-a, 타당). 차가 앞으로 간다.
- 그러나 prior가 **고속-충돌 lap1로 73% 편중**(§6)이라 출발부터 한계페이스로 가속 →
  **lap2(또는 첫 코너)에서 원본 정책과 같은 충돌을 재현**할 공산이 크다(§3: cap15 27%·cap20 11%
  즉시충돌). value 재사용은 D3(§4)로 이 경향을 *강화*. → **여전히 완주 실패하거나, 운 좋아도
  불안정 저완주율**일 가능성이 가장 높다.
- §9-1(완주-우선 단계화)을 하면: **56~60s대 완주(baseline 107s 초과, 문자적 성공·"모방")는
  현실적**(중간 확률). 이게 가장 안전한 수확.
- §9-2(D3 수정) 없이 **진짜 개선(<56s)은 낮은 확률**(value가 고속-충돌로 밀어 닫힌 루프가 못 버팀).
  D3를 고치고 warm-start까지 더하면 stretch 확률이 의미 있게 오르나, 여전히 **상위 리스크는 §3의
  한계-고속 재료가 닫힌 루프에서 지속 불가**라는 데이터 사실.

**정직한 베팅 평가**: "안전 56s 모방"은 **할 만한 베팅**(단 §9-1로 분리 검증할 것). "진짜 개선
<56s"는 **현 계획 그대로면 실패가 더 그럴듯**하고, §9-2+9-3를 추가해야 *해볼 만한* 베팅이 된다.
실패해도 §2~§5의 "충돌-위주 offline 데이터에서의 D3·closed-loop covariate-shift·한계주행 재현
실패" 분석은 정성평가 산출물로 **충분히 가치 있다**(013 §5 주장에 동의).

**판정 재확인: (B) 조건부** — §9-1·9-2를 충족하면 안전 층위는 현실적, 진짜 개선은 9-2·9-3 추가
시에만 *시도 가능*. 9-1·9-2 없이 일괄 실행은 P6와 유사한 완주 실패가 가장 그럴듯.

---

## 11. 검수자 메모 (불확실성·미검증 항목)

- 원본 lap2 충돌이 stochastic 노이즈 vs 구조적 한계인지 **완전 분리 못함**(데이터는 training=True
  stochastic rollout). deterministic 닫힌 루프가 더 나을 여지는 있으나, diffusion per-step std와
  covariate shift가 상쇄. → §3 리스크는 "높음"이되 "확정 실패"는 아님.
- §9-2(a) 충돌 penalty 증폭의 실제 효과는 **재계산으로 검증 후** 채택할 것(여기선 산술 추정만 제시).
- §9-3 warm-start의 구현 가능 범위(코어 무변경)는 sampling 글루 수준에서 가능하다고 판단하나
  **미구현·미검증**. plan_f1tenth.py에서 직전 trajectory를 초기 노이즈에 주입하는 형태 권장.
- 본 검수는 **읽기/비파괴 npz 조회만** 수행. git add/commit/push 미실행(규약 준수). 코드 무변경.
- 1차 산출물 스크립트: `/tmp/critic_data_audit.py`, `/tmp/critic_lap2_d3.py`(휘발성, 재현 가능).

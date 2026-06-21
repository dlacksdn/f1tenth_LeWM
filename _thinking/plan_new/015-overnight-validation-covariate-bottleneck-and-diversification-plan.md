# 015 — 야간 검증 결과: "마진(質)이 아니라 복구 covariate가 병목" + 데이터 다양화 계획

> 2026-06-21. [[014-cap10-quality-probe-and-overnight-validation]]의 야간 자율 검증을 **전부 실행한
> 결과**와, 그로부터 내린 **방향 결정 + 고려 중 방안 + 적대적 critic 검수 포인트**를 정리한다.
> append-only. 엄밀(수치·파일·라인) + 쉽게(표·비유·용어). git/구현 미실행(기록만).
>
> 읽는 순서: **과정(§1~2) → 결정적 발견(§3) → 그래서 내린 결정(§4) → 고민 중 방안(§5) →
> 계획(§6) → ★critic이 검수할 것 전부(§7) → 함정·비용(§8)**.

---

## 0. 한 줄
밤새 검증한 결과, **cap10(고속 한계주행)도 cap5(저속 큰 마진)도 BC로 완주 0/10**이고 차이는
**생존시간뿐**(cap10 64 step → cap5 356 step, 5.5배). 즉 **데이터 품질(마진)은 "얼마나 버티나"만
좌우하고 "완주 여부"의 원인이 아니다.** 진짜 공통 병목은 **"라인 밖으로 표류했을 때 복구하는
데이터가 없음"(복구 covariate 부재)** — 두 데이터 모두 시작 pose가 한 점에 고정되고 단일 라인만
반복하기 때문. → 결정: **cap5 폐기, cap10을 "더 많이"가 아니라 "더 다양하게(시작점 랜덤화 + 작은
노이즈 주입)" 수집해 BC 완주라는 토대를 만든다.** value(γ0.999)는 병행, obs 개선은 백업 가설.

---

## 1. 무엇을 검증했나 (014 야간 계획의 실행)
GPU 하나로 순차(알림 체이닝), 사용자 취침 중 무인 실행. 4가지:
1. **품질 워크플로**(5렌즈 병렬+적대종합): cap10 완주 30ep 데이터의 질을 다각도 실측(cap5 22ep 대조).
2. **V1(CPU)**: §6 RTG를 일관 통계로 재계산(012 오기 정정) + 충돌페널티 스윕(raw 경로 노브 탐색).
3. **V2(S4 게이트)**: cap10 prior + (기존 P6) value, scale 0.1/0.3 × K 1/3 평가.
4. **V3**: cap5-only prior 신규 학습 → cap5 BC 평가(K1/K3) = **마진 가설 결정 게이트**.

---

## 2. 맞이한 결과 (전부 `run_logs/` 보존)

### 2.1 품질 워크플로 — "질(노이즈)은 무죄, 한계주행이 결함"
| 측면 | cap10 | cap5 | 판정 |
|---|---|---|---|
| 완주 길이 일관성 | CV **0.19%**(2807~2830, 전부 2랩) | CV 0.52% | good(오히려 cap10이 더 균일) |
| 안전마진(정면) | 충돌구간 **1.26m** | 1.25m | good(아슬아슬 아님) |
| action 노이즈 | mean\|Δsteer\| **0.172** | 0.196 | cap10이 덜 노이지 |
| **코너 거동** | corr(speed,\|steer\|)**−0.03**, 코너속도비 **0.995**, rail포화 **7.76%**, lateral-demand **2.26** | 무감속이나 저속(rail 1.10%, lat-demand 0.78) | **cap10 = 마진0 한계주행** |

→ synthesis: **질(노이즈·일관성·안전마진) 무죄. "코너 무감속+rail포화 고속=오차마진0"이라는 정책
성격(質)이 결함이나, 그 결함도 cap5와의 차이는 "절대속도(9.6 vs 4.7)"뿐.** 두 데이터 공통 약점 =
**시작 pose 다양성 0(state std=0, frame-0 lidar byte-동일) + 단일 라인 반복 → 복구 covariate 부재.**

### 2.2 V1 — RTG 일관 재계산 + 페널티 노브
- γ=0.999 시작 RTG(일관): cap10완주 **207.8** > cap20충돌 **167.7/144.3**(mean/med) > cap15충돌
  **118.6/81.8** > cap5완주 **98.0**. → 013 §5 재확인: **고속충돌 > 저속안전완주**(value가 속도 보상);
  012의 cap15=61.5는 **비재현(오기)** 확정.
- **충돌페널티 스윕(γ0.999, 충돌 직전 128 윈도 시작 RTG 음전환율)**: pen −10=**0%** → −50=**88%** →
  −100=88% → −200=100%. **→ 페널티를 −10→−50으로만 키우면 충돌 윈도 88%가 음전환**(raw 충돌 경로
  S1의 value 노브 = `termination_penalty −50`, 코어 무변경 config).

### 2.3 V2(S4) — 기존 value로는 완주 0 (단 오염 약신호)
| scale×K | 완주 | 충돌 len |
|---|---|---|
| 0.1×{1,3}, 0.3×{1,3} | **전부 0/10** | 42~130 |
- ★ **한계**: P6 value는 all-data·γ0.99로 학습 → cap10 prior와 normalizer/γ **불일치 = guidance 오염**.
  "value가 무효"가 아니라 "기존 value로는 안 됨"까지만. **깨끗한 cap10-stats·γ0.999 value는 미시험.**

### 2.4 V3(cap5 BC) — ★ 마진 가설 게이트: cap5도 완주 0 (5.5배 버틸 뿐)
| prior | K | 완주 | 충돌 len median | cmd_v |
|---|---|---|---|---|
| cap10 BC | 3 | 0/10 | **64** | 9.6 |
| **cap5 BC** | 1 | **0/10** | **~285** | 4.6 |
| **cap5 BC** | 3 | **0/10** | **~356** | 4.7 |
- cap5는 cap10보다 **5.5배 오래 버팀**(일부 ep 1064 step)지만 **둘 다 0/10**, 어느 prior도 **1랩(2860
  step)조차 못 채움.**

---

## 3. ★ 결정적 발견 (과정 → 결론)
**과정**: "cap10이 BC 완주를 못 한다(012) → cap5 혼합이 범인인가?(아니오, cap10-only도 0) → 그럼
데이터 품질(質)이 나쁜가?(사용자 질문) → 품질 워크플로 + cap5 통제실험으로 검증."

**결론 1 — 마진(質)은 "생존시간"만 좌우, "완주 여부"의 원인이 아니다.**
cap5(마진 큰 저속)는 cap10(마진0 고속)보다 5.5배 버텼다 = **마진이 compounding을 늦춘다** ✓. 그러나
**cap5도 완주 0** = 마진을 키워도 BC 완주는 안 됨. → 사용자 질문 "품질이 원인?"의 답: **부분적
(생존시간엔 영향, 완주엔 무관).**

**결론 2 — 진짜 공통 병목 = 복구 covariate 부재.**
두 데이터 모두 **시작 pose 한 점 고정 + 단일 라인 반복**(품질 워크플로 실측). closed-loop에서 작은
오차로 라인 밖으로 표류하면 **그 상태에서 어떻게 복구하는지 보여주는 데이터가 0** → BC가 복구를 못
배워 발산. cap5는 느려서 천천히, cap10은 빨라서 즉발. **이건 質(노이즈)도 量(개수)도 아닌
다양성/구조(covariate) 문제.** (비유: 차선 한가운데만 30번 본 운전자는 차선을 살짝 벗어나는 순간
어떻게 돌아오는지 모른다 — 벗어난 상황을 본 적이 없으니까.)

---

## 4. 그래서 내린 결정
1. **cap5 폐기**: 느려서 baseline(107s) 못 넘고 BC도 0, value 앵커도 낮춤(013). 단 cap5 실험의
   *결과*(마진≠완주, covariate 병목 입증)는 핵심 자산으로 보존.
2. **BC 완주를 토대로 우선**(사용자 직관): prior(BC) 완주 없이는 value·stitching 다 무의미(011 §2).
3. **cap10을 "더 많이"가 아니라 "더 다양하게" 수집**: 복구 covariate를 넣어야 BC 완주가 열린다.
   "같은 시작점·같은 정책 30→100ep"는 **단일 라인만 두꺼워져 병목 그대로**(cap5도 22ep인데 0).
   - 확인됨: 시작점 고정은 **설계 결정**(`measure_gap_follower.py:27` "eval pose 고정")이지 환경 제약이
     아니다 — `env.reset(poses)`로 시작 pose 다양화 **가능**. 노이즈 주입도 `collect_episode`의 action→
     `env.step`(L148/157) 사이에 주입 지점 존재.

---

## 5. 고민 중인 방안 (covariate를 넓히는 길들 + 조합)
| 방안 | 무엇 | 위상 |
|---|---|---|
| **시작점 랜덤화** | `env.reset(poses)`로 centerline 따라 다양한 시작 pose | **메인(채택)** |
| **노이즈 주입(DART)** | action에 작은 `N(0,σ)` → 표류 후 정책 복구를 데이터화 | **메인 병행**(σ 작게, 왜곡 모니터링) |
| **깨끗한 value(γ0.999)** | 데이터 말고 *추론*에서 표류 교정 (S4 오염 정정판) | **병행 보조**(수집과 무관) |
| **DAgger** | prior 돌려 *실제 방문* 표류 상태를 cap10 정책이 라벨→추가→재학습 반복 | **백업**(가장 정확·무거움) |
| **obs/conditioning(C8)** | lidar 128→256 또는 프레임 스택(동역학) | **가설 전환 백업** |
| **S1(충돌+페널티−50)** | raw 충돌 prior + V1이 찾은 페널티 노브 | **Phase 3 옵션** |

> 사용자 의도: 시작점 랜덤화 + 노이즈 주입을 **둘 다**(노이즈가 데이터를 크게 왜곡하지 않는 선에서).

---

## 6. 계획 (Phase 게이트)
**Phase 1 — 다양화 수집(RL_project)**: cap10 정책 rollout에 ① 시작점 랜덤화(centerline 기반) ② 작은
노이즈 주입(σ 스윕, "완주율 유지되는 최대 σ" 채택). 충돌 ep는 충돌 전 복구 구간만/폐기 → 완주·복구
성공 데이터 ~50–100ep. **검증**: 수집 데이터 covariate 폭(시작 pose std·bundle width)이 기존 30ep보다
넓어졌는지 측정(워크플로 지표 재사용).

**Phase 2 — 재학습 + BC 게이트(★핵심)**: cap10_diverse로 prior 재학습(촘촘 ckpt) → BC 평가(K1/K3).
✅ 완주 나오면 = "covariate가 병목"이었음 확정 + 토대 → Phase 3. ❌ 안 나오면 → Phase 4(obs).

**Phase 3 — value stretch(병행)**: cap10-stats + γ0.999 value 학습(S4 오염 정정). BC 완주 후 scale
스윕(0/0.1/0.3) → 표류 교정 + <56s. (+옵션 S1: 충돌데이터+페널티−50.)

**Phase 4 — 백업(게이트 실패 시)**: obs/conditioning(lidar 256/프레임스택) 또는 DAgger.

---

## 7. ★ 적대적 critic이 검수할 것 (전부 — 다음 세션 1차 소스로 반박/확증)

**D1 — "복구 covariate 부재가 완주 실패의 공통 원인"이 충분히 입증됐나?**
근거 = cap5·cap10 둘 다 0/10 + 시작 pose std=0 + 단일 라인. 그러나 이는 "마진이 원인 아님"을 보일 뿐,
"covariate가 *유일/주* 원인"을 증명하진 않는다. obs/conditioning(C8)이 동시 원인일 수 있다(둘 다 실패
= 둘 다의 공통 원인이 obs일 수도). **반증 설계 없이 covariate로 단정하면 012 A5와 같은 과잉해석 위험.**

**D2 — "마진이 생존시간을 좌우"의 인과가 깨끗한가?**
cap5가 5.5배 버틴 게 "마진" 때문인가, 단지 "저속이라 같은 거리를 더 많은 step으로 가서" 인가? 거리
환산(cap5 356×4.6≈1638 vs cap10 64×9.6≈614, 2.6배)으로도 cap5가 더 갔으나, "마진"과 "저속
compounding 지연"은 사실상 같은 것이라 분리 측정이 안 됐다. **인과 표현을 신중히.**

**D3 — S4 결론의 한계 (재확인 필수)**: V2는 P6 value(all·γ0.99)라 normalizer·γ 불일치 = 오염.
"value 무효"로 읽으면 오류. **깨끗한 cap10-stats·γ0.999 value로 S4를 다시 해야 value 레버 판정 가능.**
Phase 3가 이를 수행하나, critic은 "Phase 2 BC 완주 전에는 value가 고를 prior plan이 없다"(011 §2)는
순서 의존성도 점검.

**D4 — 노이즈 주입이 cap10 한계주행에서 작동하나?**
cap10은 마진0(rail포화 7.76%, lateral-demand 2.26)이라 **작은 σ에도 즉시 충돌** → 복구 데모 대신 충돌만
양산할 위험. "완주율 유지되는 σ" 윈도가 **존재하긴 하나** 너무 좁아 covariate를 의미 있게 못 넓힐 수
있다. critic: σ 스윕에서 (완주율 유지 ∧ covariate 실측 확대)가 동시 성립하는 구간이 있는지 데이터로.

**D5 — 시작점 랜덤화의 실현성**: `env.reset(poses)`가 임의 pose에서 안정 작동하나? 트랙 밖/벽 안 pose
생성 위험(차가 시작부터 벽에 끼임). centerline+자세 정렬이 필수(`extract_centerline.py` START_POSES
검증 로직 재사용). critic: 생성 pose의 free-space·heading 정합 점검.

**D6 — 목표 정합(stretch vs 완주)**: covariate로 BC 완주해도 56s "모방"이라 stitching(<56s) 입증 아님
(013 C9). 보고서 목표가 "baseline 초과 완주"면 정당, "offline RL stitching 입증"이면 Phase 3 필수.
**무엇을 성공으로 부를지 미리 고정.**

**D7 — 자원 배분 리스크**: covariate 가설이 틀리고 obs(C8)가 진짜면, Phase 1~3에 GPU 수~수십 시간
쓰고 실패. **Phase 4를 더 앞당겨(저비용 obs 프로브) 가설을 먼저 가를 수 없나?** (예: 256 다운샘플로
빠른 BC 한 판.)

**D8 — V1 페널티 노브의 비약**: "−50에서 충돌윈도 88% 음전환"은 **offline RTG 통계**다. value가 학습으로
이를 근사하고 closed-loop에서 실제 회피로 이어지는지는 별개(013 §5.2 "RTG 변별 ≠ 주행 회피" 비약).
S1을 띄울 때 이 비약을 시험으로 닫아야.

**D9 — offline RL 정신 정합**: 시작점 랜덤화·노이즈 주입 = **새 데이터 생성**이다. "offline(고정 데이터)"
원칙과 충돌 아닌가? (방어: 011 §8.2 "캡 정책 rollout은 정공법·교수 확정", no-expert 안 깸. 단 critic은
"수집 분포를 인위 설계하는 것"이 과제 취지에 맞는지 재확인.)

**D10 — 비용·순서 견적**: Phase 1(2-4h)+2(1-2h)+3(value 5-10h) 순차(GPU 7/8GB 동시 불가). 현실적인가,
병행 주장(Phase 3가 1·2와 병행)이 GPU 단일성과 모순 아닌가.

---

## 8. 함정 · 비용 · 규약
- **GPU**: run_in_background, kill 단독 명령(복합+kill=exit144), 학습·평가 동시 불가(순차).
- **정규화 정합**: 평가는 `F1TENTH_MODE=<학습모드>`로(normalizer 자기정합); value 쓰면 prior·value 같은
  normalizer 필수(S4 오염의 교훈).
- **촘촘 체크포인트**(save_freq2000/n_saves20), 정지 전 state_N>0 확인([[verify-before-kill]]).
- **로그 보존**(`run_logs/` 삭제금지), 출력 `--out` 절대경로, V_MAX20·2랩·Oschersleben.
- **코어 무변경**(temporal/diffusion/trainer); 수집은 RL_project·글루, 시작점·노이즈도 글루.
- **no-expert**: cap10 캡 정책 rollout만(expert 금지).

## 9. 참조
- 야간계획·품질: [[014-cap10-quality-probe-and-overnight-validation]] / critic: [[013-adversarial-critic-of-012-crash-combo]]
- 선회: [[012-cap10only-bc-result-and-crash-combo-pivot]] / SSOT: [[011-staged-gate-plan-v2-finalized]]
- 로그: `run_logs/{a3_cap10_K3, s4_cap10_s*, cap5_bc_K*}.{json,log}` · `V1_rtg_penalty.txt` · `train_cap5.log`
- 코드: `f1tenth_RL_project/scripts/{collect_crash_data.py(L148/157 action주입), measure_gap_follower.py(L27 고정pose), extract_centerline.py(START_POSES)}` · 로더 `vendor/diffuser/diffuser/datasets/f1tenth.py`(cap5 모드 글루) · `config/f1tenth.py`(termination_penalty, discount)
- 데이터: `f1tenth_RL_project/runs/crash_data/`(동결)

# 020 — Offline RL 모델 선정: 검증된 후보 서베이 → TD3+BC 우선 결정 (세션 인계)

> 2026-06-21. 019(Diffuser 폐기 → 정책 기반 Offline RL 전환) 이후, **24-에이전트 리서치 워크플로우**로
> 검증된 모델 ~60개를 서베이·적대적 검증하고, 과제 PDF 원문(p6-8)을 직접 확인해 **주제1(순수 Offline RL)
> 확정 + 최우선 모델 TD3+BC**를 결정한 문서. append-only. 엄밀(인용수·근거) + 쉽게(표). git/구현
> 미실행(기록만). 다음 세션 인계용.

---

## 0. 한 줄 (가장 중요)
**과제 성공조건이 "baseline(107.16s)을 _이겨라_"이므로, occupancy-_matching_(천장=expert)인 offline IRL/모방이
아니라 reward-_maximizing_인 value 기반 Offline RL을 쓴다.** 충돌 데이터를 음의 신호로 가장 알뜰히 써먹는
것도 이 계열 → 사용자의 "데이터 아까워서 offline" 직감과 정확히 일치. **결정: 주제1(순수 Offline RL),
최우선 TD3+BC → ReBRAC → IQL → EDAC. CQL·diffusion actor·DT는 회피.** IRL은 보고서용 비교 1런(DWBC/ReCOIL).

---

## 1. 이 문서의 맥락 (019 이후)
019에서 "Diffuser 폐기, 정책 기반 Offline RL로 전환"을 결정했으나 **구체 알고리즘은 미정**이었다. 이번 세션:
1. **리서치 워크플로우**(24 에이전트, 서브토큰 ~100만, 23분): 5개 패밀리 병렬 서베이(value/시퀀스·생성형/모방·IRL/
   레이싱/2023-26 최신) → 18개 후보 적대적 검증(인용수·순수 offline 호환·50Hz 실시간) → 종합.
2. **과제 PDF 원문(p6-8) 직접 확인** → 019가 뭉뚱그린 부분 정정(§2).
3. 사용자가 "쌓은 데이터 보존" 이유로 offline 선호 표명 → 분석이 이를 뒷받침(§3-4).
4. **결정: 주제1 + TD3+BC 우선**(§5). 020 문서로 저장 지시받음.

## 2. 과제 PDF 정정 — 주제는 둘, 제약은 주제1에만 (p6-8)
개인 추가 프로젝트는 **두 주제 중 택1**이며, 019는 이를 "과제=Offline RL"로 합쳐버렸다(부분 오류). 원문:
- **주제1 — Offline RL** (p7): "약 100초대 policy로 데이터 수집 → Offline RL → **환경과 추가 상호작용 없이
  정책 개선**". 기대결과 = **기존 policy 대비 빠른 lap time**.
- **주제2 — Inverse RL** (p8): "expert 데이터로 **숨겨진 보상함수 추정** → **학습된 보상으로 RL 수행** →
  사람이 설계한 reward 없이 정책 학습". 기대결과 = **일반화·새 트랙 안정 주행·학습된 reward 효과 분석**.

→ **"추가 상호작용 없이" 제약은 주제1에만 있다.** 주제2(IRL)는 정의상 "reward 추정 → RL 수행"이라 **online이
오히려 전제**다. 따라서 "IRL은 online이어도 되는가?"의 답은 **주제2를 택하면 Yes**. 하지만 아래 §3 이유로 주제1 선택.

## 3. 결정적 통찰 — beat vs match (왜 value 기반 Offline RL인가)
| 계열 | 본질 | 성능 천장 | 충돌 데이터 활용 | baseline 격파 |
|---|---|---|---|---|
| Offline IRL/모방 (SMODICE·DemoDICE·ReCOIL·DWBC) | occupancy-**matching** | **≈ expert** | 약함(주로 expert occupancy) | 구조적으로 **불가**(맞추기) |
| Value 기반 Offline RL (TD3+BC·ReBRAC·IQL·EDAC) | reward-**maximizing** | **> demonstrator 가능** | **강함**(충돌=낮은 return→actor 밀어냄) | **가능** |

성공조건이 "이겨라"인데 occupancy-matching은 "재현"이 최적해라 baseline을 **못 넘는다**(검증에서 SMODICE
'weak', ReCOIL도 'near-expert' 한계 명시). 반면 value 계열은 critic이 충돌을 낮은 return으로 깔아 actor를 빠른
expert 라인으로 당긴다 = **"expert+충돌 혼합 활용" 요구 그 자체**. 추론은 전부 **단일 결정론적 MLP forward
1회**(Diffuser와 정반대) → 50Hz 통과, d3rlpy/CORL PyTorch 드롭인으로 기존 DreamerV3-torch 인프라에 얹힘.

## 4. 사용자 결정 근거 — "데이터 아까워서 offline"
누적 데이터의 대부분은 **충돌 궤적**(cap15=371충돌, cap20=291충돌). value 기반 offline RL은 이 충돌을 **음의
예시로 Q/가치 함수를 날카롭게** 만드는 데 가장 잘 쓴다. 즉 "쌓은 데이터 끝까지 쓰기" 목적과 정확히 부합.
(주의: IRL로 가도 expert 데이터는 reward 입력으로 쓰이나, 정책 학습 무게가 online rollout에 실려 충돌
데이터 비중↓. 그래서 "데이터 알뜰" 목적엔 offline RL이 더 맞음.) + 과거 offline 실패는 BC(가치학습 없음)·
Diffuser(부적합 planner)·dreamer-offline(그래도 9.5% 완주)였지 **제대로 된 offline RL 알고리즘은 아직 한 번도
안 써봤다** → 주제1은 "실패 예정"이 아니라 "아직 제 실력 안 낸" 길.

## 5. 추천 스택 (주제1 — 순수 offline, 검증된 모델)
### 5.1 액션 가능한 shortlist
| 순위 | 모델 | 연도/학회 | 인용수(검증 Jun 2026) | 왜 / 구현 |
|---|---|---|---|---|
| **1 (먼저)** | **TD3+BC** | NeurIPS 2021 Spotlight | ~982(SS)/~1.5k(GS) | 최저 리스크·최고 가성비. value-weighted BC로 expert+충돌 활용(BC 단독 불가). **이 F1TENTH 벤치마크 검증**(arXiv:2408.04198). TD3에 2줄 추가. 구현: sfujim/TD3_BC, d3rlpy `TD3PlusBC`, CORL 단일파일. |
| 2 (성능 극대화) | **ReBRAC** | NeurIPS 2023 | ~100+ (SS ~128) | TD3+BC의 현대화 SOTA(decoupled actor/critic BC penalty, deeper net, critic LayerNorm, larger batch). 51개 D4RL/V-D4RL ensemble-free SOTA. BC penalty 노브로 baseline 넘기기 푸시. 구현: corl-team/ReBRAC(JAX)+CORL PyTorch. |
| 3 (안정 앵커) | **IQL** | ICLR 2022 | ~1,521(SS)/~2k(GS) | in-sample expectile → OOD 안 건드려 **충돌 데이터에서 가장 안정**(CQL식 blow-up 회피). F1TENTH 완주 검증. ※**plain IQL만**, IDQL(diffusion)은 금지(Diffuser 실패 재현). 구현: ikostrikov, d3rlpy, CORL. |
| 4 (대안 보수) | **EDAC** | NeurIPS 2021 | ~402(SS)/~557(GS) | 앙상블 불확실성 비관 — **suboptimal+expert 혼합이 정확히 강점**. 앙상블은 학습 전용, 추론은 단일 MLP. 학습 비용↑(N~10-50, eta 민감). 구현: snu-mllab/EDAC, CORL. |
| 5 (비교 ablation) | **DWBC / ReCOIL** | ICML 2022 / ICLR 2024 Spotlight | ~80 / ~60 | offline-IL의 정직한 답·대조군. occupancy-matching이라 **천장 ≈expert(못 이김)**지만, "matching vs maximizing" 보고서 분석축. 구현: ryanxhr/DWBC, hari-sikchi/Dual-RL. |

### 5.2 ★ TD3+BC를 ReBRAC보다 "먼저" 두는 이유 (사용자 질문 반영)
**ReBRAC가 더 진보·고성능인 건 맞다**(천장이 더 높음). 하지만 추천은 "어느 게 더 우월한가"가 아니라
**"어느 순서로 가는가"**다. ReBRAC를 _나중_에 두는 이유:
1. **튜닝 부담**: ReBRAC의 SOTA 수치는 **수천 회 튜닝**의 산물. 성능을 좌우하는 BC penalty 계수가 **2개**(actor/
   critic 분리)라 새 데이터마다 재튜닝 필요. TD3+BC는 노브가 사실상 **1개(alpha)**, 기본값에서 잘 돈다 →
   **튜닝 레시피 없는 우리 f1tenth 데이터에선 TD3+BC가 "그냥 작동"할 확률이 훨씬 높고, 잘못 튜닝한 ReBRAC는
   기본 TD3+BC보다 못할 수도 있다.**
2. **ReBRAC = TD3+BC + 개선 스택**: ReBRAC는 TD3+BC 위에 (분리 BC penalty·깊은 net·critic LayerNorm·큰 배치·
   긴 학습)을 얹은 것. 즉 **TD3+BC를 먼저 세우면 그게 ReBRAC의 출발 baseline**이 되고, 각 개선이 우리 데이터에서
   몇 % 이득인지 측정 가능 → 좋은 엔지니어링 + **보고서 ablation**("TD3+BC → +ReBRAC tricks → +X% lap time").
3. **성숙도·검증**: TD3+BC ~982인용·4년+ 커뮤니티 검증·레퍼런스 구현 3종. ReBRAC ~100인용·2023·공식 JAX.
   → 놀랄 일이 적음.
4. **진단 가치**: TD3+BC가 먼저 돌면 파이프라인 전체(reward 라벨링·로더·eval gate)를 **가장 단순한 방법으로
   검증** → 이후 ReBRAC 전환 시 "알고리즘이 도왔나"만 격리. ReBRAC부터 시작해 실패하면 알고리즘/튜닝/파이프라인
   중 무엇 탓인지 모름.

**요약**: ReBRAC = 더 높은 천장 + 더 높은 튜닝비용/분산. TD3+BC = 약간 낮은 천장 + 훨씬 낮은 리스크·노력 +
거의 보장된 합리적 결과. **"baseline 넘을 수 있나"를 먼저 TD3+BC로 확인 → 최종 수치는 ReBRAC로 극대화.**
둘 다 돌려서 보고서에 같이 싣는 게 이상적(ablation).

## 6. 회피 목록 (전부 근거 있음)
- **CQL** (NeurIPS'20, ~2.6k인용): **이 F1TENTH 벤치마크에서 conservative loss blow-up·하이퍼 초민감**
  (arXiv:2408.04198) → 1차 후보에서 제외, 비교용으로만.
- **모든 diffusion actor**: IDQL(N*T 샘플링=**unsuitable**), Decision Diffuser(~0.4Hz, ~125배 느림=**unsuitable**),
  Diffusion-QL(N~5 denoising=**borderline**) → Diffuser 실패(0% 완주) 재현 위험. one-step 증류형 **DTQL**(NeurIPS'24)만
  예외적으로 실시간 통과하나 신규·검증 부족.
- **Decision Transformer** (~2k인용): cross-track 일반화는 최고지만 **stitching 약해 데이터를 못 이김**(return
  conditioning이 baseline 격파에 부적합). EDT/Q-Transformer는 그나마 stitching 보강.
- **Trajectory Transformer·Diffuser·AdaptDiffuser·Decision Stacks**: beam-search/denoising **open-loop planner =
  실시간 실격**.

## 7. IRL 정산 (보고서용, §2 주제2와 연결)
워크플로우 검증 종합(원문 보존):
- **고전 IRL(GAIL ~3.4k·AIRL ~710·GCL·MaxEnt-IRL ~3k·f-IRL·OPIRL)** = **전부 online 전용**(discriminator/reward
  학습에 env rollout) → 주제1 offline 제약 **실격**. (주제2를 택했다면 합법이었음.)
- **순수 offline IRL/모방** 존재·검증됨: DICE류(SMODICE·DemoDICE·LobsDICE)·DWBC·ReCOIL·**CLARE**(offline
  model-based reward 복원)·IQ-Learn offline모드. 전부 무상호작용·단일 forward. **단 occupancy-matching →
  천장 ≈expert(beat 목표와 구조적 불일치).**
- **혼합데이터 주의**: vanilla IQ-Learn은 충돌혼합에 약함(후속 SubIQ/UNIQ). ReCOIL/DWBC/DemoDICE가 expert+
  suboptimal 혼합 전용으로 신뢰 가능한 offline-IRL 픽.
- **cross-map reward 분석** 원하면: offline에서 전이가능 reward를 복원하는 유일 경로는 **CLARE**(model-based,
  무겁지만 기존 Dreamer fork 활용) 또는 **OTR**(optimal-transport reward 라벨링 → offline-RL 정책에 주입).
- **NET**: offline IRL은 가능하고 **DWBC나 ReCOIL 1런을 비교 ablation**으로 돌리면 "matching(못 이김) vs
  maximizing(이김)" 대조가 강력한 보고서 축. 단 헤드라인 방법으로는 reward-maximizing offline RL에 지배됨.

## 8. 검증된 모델 전체 풍경 (~60개, 요청 "최대한 많이" — 보존)
> 워크플로우가 인용수·online/offline·실시간(50Hz·1패스)·적합도까지 검증. 삭제 금지(보고서 자료).

| Method | Year | Venue | ~Citations | On/Offline | Real-time(50Hz,1pass)? | Fit |
|---|---|---|---|---|---|---|
| **TD3+BC** | 2021 | NeurIPS'21 Spotlight | ~982 SS/~1.5k GS | Pure-offline | Yes(det MLP) | **STRONG(top)** value-weighted BC, 최단 인프라 |
| **ReBRAC** | 2023 | NeurIPS'23 | ~56-128 | Pure-offline | Yes(det MLP) | **STRONG** TD3+BC 현대화 SOTA |
| **IQL** | 2021/22 | ICLR'22 | ~1,521 SS/~2k GS | Pure-offline | Yes(AWR MLP) | **STRONG** 충돌데이터 최안정, F1TENTH검증 |
| **EDAC** | 2021 | NeurIPS'21 | ~402 SS/~557 GS | Pure-offline | Yes(앙상블=학습전용) | **STRONG** suboptimal혼합 강점, 학습 무거움 |
| SAC-N | 2021 | NeurIPS'21 | ~400-500 | Pure-offline | Yes(큰 앙상블 학습전용) | Good, N커서 EDAC보다 무거움 |
| CQL | 2020 | NeurIPS'20 | ~2,595 SS/~3k GS | Pure-offline | Yes | MODERATE 이 벤치마크서 불안정 |
| Cal-QL | 2023 | NeurIPS'23 Spotlight | ~153-313 | Conditional | Yes | MODERATE offline-only면 CQL로 환원 |
| MCQ | 2022 | NeurIPS'22 | ~116-161 | Pure-offline | Yes | Good, CQL보다 덜 보수적 |
| SAC-RND | 2023 | ICML'23 | ~70-90 | Pure-offline | Yes | Good, ensemble-free 보수성 |
| XQL | 2023 | ICLR'23 Oral | ~130 | Offline-capable | Yes | Good IQL-family, Gumbel loss 민감 |
| SQL/EQL(IVR) | 2023 | ICLR'23 Spotlight | ~132 | Pure-offline | Yes | Good in-sample, 노이즈 robust |
| BCQ | 2019 | ICML'19 | ~2,624 GS | Pure-offline | Borderline(~100 VAE샘플/step) | MODERATE 1세대, TD3+BC에 밀림 |
| BEAR | 2019 | NeurIPS'19 | ~941 | Pure-offline | Yes | Weak-modern, MMD finicky |
| AWAC | 2020 | NeurIPS'20 WS | ~430+ (~1k?) | Conditional | Yes | Strong-ish, IQL offline보다 약간 아래 |
| AWR | 2019 | arXiv'19 | ~400-600 | Offline-capable | Yes | Baseline, AWAC/IQL에 대체됨 |
| OneStep-RL | 2021 | NeurIPS'21 | ~300-400 | Pure-offline | Yes | Robust baseline, 천장 낮음 |
| Fisher-BRC | 2021 | ICML'21 | ~300+ | Pure-offline | Yes | Solid, 구현 적음 |
| PLAS(+P) | 2020 | CoRL'20 | ~250-350 | Pure-offline | Yes(VAE decoder 1패스) | Good 2차, F1TENTH dynamics-robust |
| PBRL | 2022 | ICLR'22 | ~150-200 | Pure-offline | Yes(앙상블 학습전용) | 비교용, 이론적·무거움 |
| LB-SAC | 2022 | NeurIPS'22 WS | ~30-50 | Pure-offline | Yes | EDAC/SAC-N 학습효율 변형 |
| MOPO | 2020 | NeurIPS'20 | ~799 SS/~1.3k GS | Pure-offline | Yes(model 학습전용) | MODERATE model-based, 고속 dynamics 리스크 |
| MOReL | 2020 | NeurIPS'20 | ~600-700 | Pure-offline | Yes | 비교용, HALT 보수적 |
| COMBO | 2021 | NeurIPS'21 | ~500-600 | Pure-offline | Yes | Advanced CQL+model, 구현 난도↑ |
| RAMBO | 2022 | NeurIPS'22 | ~150-200 | Pure-offline | Yes | 비교용, adversarial model 불안정 |
| MOBILE | 2023 | ICML'23 | ~70-99 | Pure-offline | Yes(model 학습전용) | **STRONG-model-based** SOTA·혼합데이터, dynamics 리스크 |
| Decision Transformer | 2021 | NeurIPS'21 | ~1,951-2,327 SS | Pure-offline | Moderate(context K 1패스) | MODERATE cross-track 최강이나 stitch 약→못 이김 |
| Elastic DT | 2023 | NeurIPS'23 | ~26+ | Pure-offline | Yes(1 TF pass) | Good DT변형, 혼합데이터 stitch 보강 |
| Q-Transformer | 2023 | CoRL'23 | ~300-450 | Offline-capable | Near-RT(autoreg) | Interesting, 이산화로 정밀도 제한 |
| Online DT | 2022 | ICML'22 Oral | ~400-600 | Hybrid | Yes | 낮은 우선순위, online finetune 금지→DT로 환원 |
| RCDTP | 2024 | arXiv 2408.04198 | <15 | Pure-offline | Yes(tree lookup) | 싸구려 baseline, single-track 최고·cross 약 |
| Trajectory Transformer | 2021 | NeurIPS'21 | ~900-1000 | Pure-offline | **NO**(beam search) | Avoid, 느린 open-loop planner |
| DTQL | 2024 | NeurIPS'24 | ~42 | Pure-offline | Yes(one-step 증류) | **STRONG diffusion-family** denoising 없이 multimodal, 신규 |
| Diffusion-QL | 2022 | ICLR'23 | ~648 SS | Pure-offline | Borderline(N~5 denoising) | MODERATE 정책이나 반복추론→DTQL/EDP 권장 |
| IDQL | 2023 | arXiv/EECS TR | ~306 SS | Pure-offline | **NO**(N*T 샘플링) | UNSUITABLE Diffuser 실패 재현 |
| QGPO | 2023 | ICML'23 | ~157 | Pure-offline | **NO**(energy-guided) | RT부적합 |
| SfBC | 2022 | ICLR'23 | ~150-200 | Pure-offline | **NO**(K샘플/step) | RT부적합 |
| Diffusion Policy | 2023 | RSS'23/IJRR'24 | ~2,160 SS | Offline-capable(BC) | **NO**(chunked denoising) | 낮은 우선순위, 충돌데이터 미활용 |
| Diffuser | 2022 | ICML'22 | ~958 SS | Pure-offline | **NO**(수백 denoising) | **AVOID** — 0% 완주로 실패한 그 planner |
| Decision Diffuser | 2023 | ICLR'23 Oral | ~635 SS | Pure-offline | **NO**(~0.4Hz) | UNSUITABLE 생성 planner |
| AdaptDiffuser | 2023 | ICML'23 Oral | ~150-250 | Offline-capable | **NO**(Diffuser식) | Avoid |
| Decision Stacks | 2023 | NeurIPS'23 | ~50-120 | Pure-offline | **NO**(모듈 생성 planning) | RT부적합 |
| BC | 1988/2010s | ALVINN 등 | thousands | Pure-offline | Yes | Baseline/정책head, 충돌 평균화·covariate |
| **DWBC** | 2022 | ICML'22 | ~80 | Pure-offline | Yes(weighted BC) | STRONG offline-IL, 충돌 downweight, 천장≈expert |
| DemoDICE | 2022 | ICLR'22 | ~90 | Pure-offline | Yes | Strong offline-IL, 혼합용, matching 천장 |
| SMODICE | 2022 | ICML'22 | ~30-65 | Pure-offline | Yes | WEAK(beat목표), reward-free·충돌 버림 |
| LobsDICE | 2022 | NeurIPS'22 | ~40 | Pure-offline | Yes | offline-IL(state-only), 동일 천장 |
| **ReCOIL(Dual-RL)** | 2023/24 | ICLR'24 Spotlight | ~56-69 | Pure-offline | Yes(IQL기반 MLP) | STRONG offline-IL, discriminator-free·임의혼합, near-expert |
| CLARE | 2023 | ICLR'23 | ~20-60 | Pure-offline | Yes(model 학습전용) | 유일 OFFLINE IRL(reward복원·cross-map), 무겁·Dreamer활용 |
| OTR | 2023 | ICLR'23 | ~60+ | Pure-offline | Yes(downstream IQL/BC) | Good bridge, OT reward라벨→offline RL |
| IQ-Learn | 2021 | NeurIPS'21 Spotlight | ~159-200(~400+ GS?) | Conditional | Yes(SAC 1패스) | MODERATE offline모드=CQL유사, 혼합 약(SubIQ/UNIQ) |
| ValueDICE | 2020 | ICLR'20 | ~195 | Offline-capable | Yes | Borderline, offline서 BC로 붕괴 가능 |
| SQIL | 2020 | ICLR'20 | ~50-600 | Online(offline DIY) | Yes | 대부분 실격, online 0-reward 수집 |
| DAgger | 2011 | AISTATS'11 | ~6000+ | Online | Yes | 실격, 대화형 expert 질의 |
| GAIL | 2016 | NeurIPS'16 | ~3,400 SS/~6k GS | Online | Yes | 실격, on-policy rollout |
| AIRL | 2018 | ICLR'18 | ~710 SS/~2k GS | Online | Yes | 실격(전이 reward지만 not offline) |
| GCL | 2016 | ICML'16 | ~1,700 | Online | Yes | 실격, env rollout 교차 |
| MaxEnt-IRL | 2008 | AAAI'08 | ~3,050 SS/~6k GS | Online | Slow(반복 MDP풀이) | 실격, online+open-loop planner |
| f-IRL | 2020 | CoRL'20 | ~70 SS/~250 GS | Online | Yes | 실격, agent marginal env 샘플 |
| OPIRL | 2022 | ICRA'22 | ~20-50 | Online | Yes | 실격, off-policy≠offline |
| GT Sophy/QR-SAC | 2022 | **Nature 2022** | ~481 SS | Online | Yes | UNSUITABLE 대규모 online(패러다임 증명만) |
| GT Sophy vision | 2024/25 | RLC'24/arXiv | <30/<10 | Online | Yes | 도메인 증명만 |
| TAL(Trajectory-aided DRL) | 2023 | IEEE RA-L | ~40-70 | Online | Yes(소형 MLP/LiDAR) | 온플랫폼 영감, racing-line reward→offline에 주입 가능 |
| RL-beats-MPC F1TENTH | 2025 | arXiv 2504.02420 | <15 | Online | Yes | SOTA 도메인 증명(online) |
| DeepRacer | 2020 | ICRA'20 | ~250-350 | Online | Yes(소형 CNN) | 도메인 인용(online·이산 많음) |

## 9. 선결과제 & 다음 액션
1. **★ reward 라벨링 (1순위 선결)**: TD3+BC/ReBRAC/IQL/EDAC 모두 **per-step reward** 필요. 수집 데이터에
   reward가 저장돼 있는지 확인(수집 정책이 dreamer라 RL reward가 있을 가능성). 없으면 **progress(센터라인 진행)
   − time + 충돌 패널티** dense reward 정의. 이게 구현 전 가장 먼저 확정할 것.
2. **TD3+BC 구현/통합**: d3rlpy `TD3PlusBC` 또는 CORL 단일파일 → 기존 데이터 로더에 드롭인.
3. **평가**: 학습 정책 → f110 Oschersleben 2랩 → lap time vs baseline 107.16s (RL_project `.venv`, V_MAX20).
4. **에스컬레이션/ablation**: TD3+BC 작동 확인 후 ReBRAC, 필요시 IQL(안정)·EDAC(혼합강점). DWBC/ReCOIL 1런으로
   matching vs maximizing 대조.

## 10. 자산·규칙 (019 계승 + 갱신)
- 자산: 019 §6 그대로(데이터 동결·dreamer 인프라·diffuser 보존·run_logs·_thinking). + **본 워크플로우 산출물**
  (검증된 ~60모델 표·shortlist) = `tasks/w9xck8wzv.output`(보고서 자료, §8에 핵심 보존).
- ★ git add/commit/push/pull·코드 구현·_thinking 저장은 **사용자 지시 시에만**(이 020 저장은 지시받음).
- ★ 로그·모델·데이터 폐기 금지.
- ★ **다른 모델 ↔ Dreamer 독립**(CLAUDE.md 갱신: "디퓨저"→"다른 모델"로 일반화). offline RL 구현 시 dreamer
  공용 코드 건드리면 **하위호환+스모크 검증** 필수. env·데이터 로더 재사용은 독립 lane으로.
- ★ GPU=run_in_background, kill 단독, 정지 전 state_N>0 확인. 평가=RL_project `.venv`·V_MAX20·2랩·Oschersleben·
  baseline 107.16s. _thinking append-only·지시 시만 읽기/저장. **한글 답변.**

## 11. 참조
- 직전: [[019-diffuser-unfit-pivot-to-policy-offline-rl]] (Diffuser 폐기·전환 결정)
- 워크플로우 산출물: `tasks/w9xck8wzv.output` (24 에이전트, 검증된 모델 표 전체)
- F1TENTH offline RL 벤치마크 근거: **arXiv:2408.04198** (TD3+BC·IQL 완주 검증, CQL blow-up 보고)
- 과제: `_thinking/raws/AIE4003_RL_F1TENTH.pdf` p6=주제 택1, p7=주제1 Offline RL, p8=주제2 Inverse RL

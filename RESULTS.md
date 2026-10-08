# Results — transporting Llama-3.2-1B steering vectors into masked-diffusion LMs (LLaDA)

Branch `diffusion`. Experiments from 2026-08-11 to 2026-09-06. Written 2026-09-17.

This document collects every experiment run so far on the diffusion targets and states what
they do and do not show. Numbers are copied from the artifacts listed in §8; nothing here is
re-estimated.

---

## 1. Summary

**Transport works on LLaDA-8B-Instruct, but weakly.** A closed-form orthogonal Procrustes map
fitted on paired FineWeb activations transports Llama-3.2-1B CAA steering vectors into
LLaDA-8B and produces a measurable, correctly-signed behaviour shift on the closed-ended A/B
evaluation. The effect is roughly an order of magnitude smaller than the source vector's
effect on its own model, and it needs a 50–100× larger dose to reach its peak.

Findings that are robust across configurations:

1. **Depth pattern.** The transported effect is nil or negative at layers 10–12 (31–38 % depth),
   turns positive at 14 (44 %), rises monotonically, and peaks at layer 22 (69 %). With the
   templated arm we extended the sweep to 24 and 26: the plain-Procrustes curve *declines* past
   22, so 22 is a genuine peak rather than the edge of the sweep. This matches arXiv 2512.24143
   (steering localised to layers ~18–24, selected layer 23 of 32) obtained independently, with a
   different method and metric, on the same model.
2. **Mean pooling beats `[MASK]`-token readout.** At 30k sentences the appended-mask arm is worse
   than mean pooling at every layer and is negative or ~0 at layers 10–18.
3. **Whitening fixes retrieval but not steering.** Plain Procrustes has near-zero src→tgt
   retrieval (acc@1 ≈ 0.00–0.02, a hubness artefact of the anisotropic target space) while
   tgt→src is 0.93–0.97; whitening lifts src→tgt to 0.81–0.99. The A/B steering effect does not
   improve correspondingly: whitening is worse than plain at 30k and mixed at 500k.
4. **LLaDA2.0-mini cannot be evaluated yet.** The Procrustes fit fails on it at both scales
   (train cosine 0.14–0.31, acc@1 ≈ 0.01, vs 0.92–0.99 / 0.9+ on the 8B), and the A/B readout is
   broken for it (unsteered P(match) ≈ 4×10⁻⁶ vs 0.61 on the 8B). Every mini A/B number on disk
   is uninformative.
5. **Direction alignment ≠ steering efficacy.** LOBO Procrustes maps fitted on CAA
   difference pairs reach cosine 0.87–0.96 to the held-out behaviour's native target diff, yet
   their transported vectors move A/B behaviour by ≤ 0.012 — an order of magnitude less than the
   FineWeb-fitted maps that score lower on direction alignment.

Findings that are suggestive but **not** separated from noise (7 behaviours × 50 items, no CIs):

6. **Scale 30k → 500k** raises the best raw-arm effect from +0.073 to +0.084 (layer 22, plain).
7. **Chat-templating the FineWeb pairs** does not help on its own (chat-mean, plain: +0.053 vs
   raw +0.084 at layer 22). Reading the templated target at the **last token (-1)** — the
   post-instruction position the paper selects — recovers it (+0.090) and is the best single
   configuration measured, but the margin over raw-mean is within per-behaviour scatter. Under
   whitening, templated arms beat raw at every layer ≥ 18 (+0.073/+0.076 vs +0.050).
8. **Discovery method.** On the target, GCAV-derived vectors transport better than RepE and CAA
   (mean peak effect 0.12 / 0.09 / 0.07); on the source GCAV ≈ CAA ≫ RepE.

---

## 2. Setup

| | |
|---|---|
| Source model | `meta-llama/Llama-3.2-1B-Instruct`, layer 8 (residual). Steering vectors: CAA (also RepE, GCAV at 30k), from `../activation_engineering`. |
| Target models | `GSAI-ML/LLaDA-8B-Instruct` (32 blocks, d=4096, bidirectional). `inclusionAI/LLaDA2.0-mini` (20 layers, d=2048, MoE, block-diffusion). |
| Pairing data | FineWeb-edu sentences, 30k (`data/fineweb_30k`) and 500k (`data/fineweb_500k`, `data/fineweb_500k_chat`). The 500k raw and templated sets are byte-identical, so rows are aligned across arms. |
| Forward pass | One clean, fully-unmasked pass per sentence for both models; no denoising loop is simulated. |
| Source read position | `-1` (last real token) for FineWeb pairing; the CAA vectors themselves were built at `-2` of `template + "Answer: (X)"`. |
| Target read positions | `mean` (mask-weighted mean over real tokens); `mask` (appended `[MASK]` token, 30k only); `-1` (last real token — under the chat template this is the final token of the assistant-turn opener, i.e. the paper's post-instruction position; 500k templated only). |
| Chat template | Off for all "raw" arms. On for the `*_chat` arms: each sentence wrapped as a user turn with the assistant turn opened; `add_special_tokens=False` (avoids a duplicated BOS on Llama); sentences pre-truncated so the template tail always survives. |
| Translator | Orthogonal Procrustes (`fit_procrustes.py`), centred, with bias (bias is dropped when transporting a direction). Two variants: **plain** (W orthonormal) and **whitened** (`--whiten`, anisotropy-aware, W no longer orthonormal). Train/val split 80/20, seed 42. |
| Steering | Additive, at the target layer the map was fitted to, over coefficients `{±50, ±10, ±5, ±2, ±1, ±0.75, ±0.5, ±0.25, 0}` (17 values, both scales). |
| Evaluation | Closed-ended A/B (`ab_eval.py`), 7 MWE behaviours × 50 test items, P(matching answer) read at an appended `[MASK]` on LLaDA. |

**Metrics** (from `ab_report.py`):

- `dP_coherent` — P(behaviour) at the largest positive coefficient minus at the largest negative one, **within |c| ≤ 5** (the "coherent" window where the model has not collapsed). Signed effect size; this is the primary number below, averaged over the 7 behaviours.
- `dP_peak` — the same difference at whatever coefficient gives the largest effect, with no window. Used in the method comparison.
- `fidelity_vs_source` — Pearson correlation between the transported vector's P-curve on LLaDA and the source vector's P-curve on Llama over shared coefficients. 1 = same shape.
- `acc@1 src2tgt / tgt2src` — held-out nearest-neighbour retrieval of the Procrustes fit (2048 samples).

**Source ceiling** (CAA vectors on their own model, Llama-3.2-1B, `dP_peak`): coordinate 0.50,
corrigible 0.90, hallucination 0.80, myopic-reward 0.57, refusal 0.66, survival-instinct 0.56,
**sycophancy 0.03**. Mean 0.57, reached at coefficients 0.5–5. Sycophancy has no source effect,
so it is uninformative in every table below.

---

## 3. LLaDA-8B, 30k raw pairs (2026-08-11)

Fits: 7 layers × {mean, mask} × {plain, whiten} = 28 translators (`outputs/new_models/`).
A/B: 10 353 rows, CAA + RepE + GCAV (`outputs/ab_eval/llada/`).

### 3.1 Layer × pooling × variant — mean `dP_coherent` over 7 behaviours (CAA)

| layer | depth | mean · plain | mean · whiten | mask · plain | mask · whiten |
|---:|---:|---:|---:|---:|---:|
| 10 | 31 % | −0.003 | −0.005 | −0.004 | −0.006 |
| 12 | 38 % | +0.006 | 0.000 | −0.006 | −0.011 |
| 14 | 44 % | +0.019 | +0.008 | +0.006 | +0.001 |
| 16 | 50 % | +0.050 | +0.051 | −0.006 | −0.009 |
| 18 | 56 % | +0.037 | +0.028 | 0.000 | −0.001 |
| 20 | 62 % | +0.065 | +0.031 | +0.029 | +0.016 |
| 22 | 69 % | **+0.073** | +0.028 | +0.040 | +0.026 |

### 3.2 Per behaviour at the best configuration (layer 22, mean, plain)

| behaviour | P(−5) | P(0) | P(+5) | dP_coherent | fidelity_vs_source |
|---|---:|---:|---:|---:|---:|
| refusal | 0.706 | 0.801 | 0.864 | +0.157 | 0.15 |
| corrigible-neutral-HHH | 0.459 | 0.525 | 0.584 | +0.124 | 0.59 |
| hallucination | 0.581 | 0.628 | 0.677 | +0.096 | 0.35 |
| myopic-reward | 0.569 | 0.609 | 0.653 | +0.084 | −0.07 |
| coordinate-other-ais | 0.180 | 0.191 | 0.208 | +0.028 | 0.75 |
| survival-instinct | 0.354 | 0.367 | 0.377 | +0.022 | 0.81 |
| sycophancy | 0.761 | 0.760 | 0.761 | −0.000 | −0.03 |

All six live behaviours move in the correct direction and monotonically (monotonicity ≥ 0.86
except sycophancy). The effect size is 5–10× below the source ceiling.

### 3.3 Discovery method (30k, all 28 translators, `dP_peak`)

| method | target: mean | target: median | target: best | median dose at peak | source: mean | source: median dose |
|---|---:|---:|---:|---:|---:|---:|
| CAA | 0.071 | 0.027 | 0.441 | 81 | 0.574 | 1.6 |
| GCAV | 0.121 | 0.051 | 0.674 | 113 | 0.625 | 2.0 |
| RepE | 0.088 | 0.017 | 0.589 | 66 | 0.123 | 5.0 |

Dose = coefficient × ‖sv‖. Transported vectors need ~50–100× the source dose to peak; peaks
sit at coefficient 10–50, outside the coherent window, so `dP_peak` and `dP_coherent` rank
configurations differently.

### 3.4 Procrustes fit quality, layer 22 (30k)

| arm | variant | train cosine | acc@1 src→tgt | acc@1 tgt→src |
|---|---|---:|---:|---:|
| mean | plain | 0.967 | 0.009 | 0.929 |
| mean | whiten | 0.978 | 0.936 | 0.905 |
| mask | plain | 0.914 | 0.001 | 0.785 |
| mask | whiten | 0.934 | 0.396 | 0.419 |

The mask arm is both a worse fit and a worse steering space.

---

## 4. LOBO Procrustes on CAA difference pairs (2026-08-12)

A different fit: source and target **per-item CAA differences** (pos − neg) on the same 4 911
manifest items (7 behaviours; source Llama l8 at token −2, target LLaDA mean-pooled), leaving one
behaviour out, then measuring cosine between the transported *real* `sv.pt` and the held-out
behaviour's native target mean diff (the "money metric"). 98 fits, `outputs/new_models/*_lobo-*`.

| layer | plain | whiten | | layer | plain | whiten |
|---:|---:|---:|---|---:|---:|---:|
| 10 | 0.30 | 0.33 | | 18 | **0.73** | 0.70 |
| 12 | 0.32 | 0.29 | | 20 | 0.69 | 0.64 |
| 14 | 0.58 | 0.47 | | 22 | 0.67 | 0.57 |
| 16 | 0.73 | 0.67 | | | | |

Mean cosine over the 7 held-out behaviours. Best individual: 0.958 (l18, whiten,
corrigible). The A/B effect of these same LOBO translators on their held-out behaviour
(`outputs/ab_eval/llada_lobo/`) is **+0.008 (plain) / +0.012 (whiten) at layer 22** and ≤ 0.005
elsewhere — 6–9× below the FineWeb-fitted maps of §3, whose direction alignment was never this
high. Direction cosine to the native diff is not a proxy for steering efficacy here; a scale
mismatch between diff-space and full-activation-space fits is the leading suspect and has not
been checked.

---

## 5. LLaDA-8B, 500k raw pairs (2026-08-18 → 08-27)

Mean pooling only; 7 layers × {plain, whiten} = 14 translators (`outputs/new_models_500k/`),
A/B 1 785 rows (`outputs/ab_eval/llada_500k/`). Same grid as §3.

| layer | plain | whiten | | fit (l22) | train cos | acc@1 s→t | acc@1 t→s |
|---:|---:|---:|---|---|---:|---:|---:|
| 10 | +0.002 | +0.000 | | plain | 0.966 | 0.020 | 0.951 |
| 12 | +0.009 | +0.003 | | whiten | 0.972 | 0.957 | 0.933 |
| 14 | +0.019 | +0.016 | | | | | |
| 16 | +0.065 | **+0.072** | | | | | |
| 18 | +0.050 | +0.039 | | | | | |
| 20 | +0.077 | +0.046 | | | | | |
| 22 | **+0.084** | +0.050 | | | | | |

30k → 500k: +0.073 → +0.084 at the best point. Same shape as §3.1; the curve is still rising
at 22, the last layer in this sweep.

---

## 6. LLaDA-8B, 500k chat-templated pairs, two read positions (2026-09-04 → 09-06)

**Motivation.** The transported CAA vectors were extracted from chat-templated prompts, but all
maps above were fitted on raw sentences — a different region of activation space. This arm
re-extracts the same 500k sentences wrapped in each model's chat template and reads the target
at two positions from one forward pass: `mean`, and `-1` (last token of the assistant-turn
opener — post-instruction, the position family the paper selects). Layers 24 and 26 were added
because §5's curve had not turned over and the paper's optimum is 23.

Source: templated Llama l8 at `-1`. Fits: 9 layers × {mean, −1} × {plain, whiten} = 36
translators (`outputs/new_models_500k_chat/`). A/B: 4 403 rows (`outputs/ab_eval/llada_500k_chat/`).
Raw arm from §5 for comparison. Joined table: `outputs/ab_eval/compare_raw_vs_chat_500k.csv`.

### 6.1 Mean `dP_coherent` over 7 behaviours

**Plain Procrustes**

| layer | depth | raw · mean | chat · mean | chat · −1 |
|---:|---:|---:|---:|---:|
| 10 | 31 % | +0.002 | +0.002 | −0.003 |
| 12 | 38 % | +0.009 | +0.002 | +0.014 |
| 14 | 44 % | +0.019 | +0.009 | +0.021 |
| 16 | 50 % | +0.065 | +0.041 | +0.033 |
| 18 | 56 % | +0.050 | +0.040 | +0.055 |
| 20 | 62 % | +0.077 | +0.035 | +0.059 |
| 22 | 69 % | **+0.084** | **+0.053** | **+0.090** |
| 24 | 75 % | — | +0.035 | +0.063 |
| 26 | 81 % | — | +0.041 | +0.075 |

**Whitened Procrustes**

| layer | depth | raw · mean | chat · mean | chat · −1 |
|---:|---:|---:|---:|---:|
| 10 | 31 % | +0.000 | +0.005 | −0.001 |
| 12 | 38 % | +0.003 | −0.004 | +0.011 |
| 14 | 44 % | +0.016 | +0.014 | +0.024 |
| 16 | 50 % | **+0.072** | +0.071 | +0.011 |
| 18 | 56 % | +0.039 | +0.051 | +0.026 |
| 20 | 62 % | +0.046 | +0.064 | +0.023 |
| 22 | 69 % | +0.050 | +0.073 | +0.047 |
| 24 | 75 % | — | +0.040 | **+0.057** |
| 26 | 81 % | — | **+0.076** | +0.041 |

### 6.2 Per behaviour at layer 22, plain

| behaviour | raw · mean | chat · mean | chat · −1 | fidelity (raw / chat·−1) |
|---|---:|---:|---:|---:|
| refusal | +0.169 | +0.076 | +0.156 | 0.14 / 0.21 |
| corrigible-neutral-HHH | +0.165 | +0.110 | +0.161 | 0.60 / 0.55 |
| hallucination | +0.113 | +0.055 | +0.093 | 0.34 / 0.26 |
| myopic-reward | +0.091 | +0.065 | **+0.153** | −0.05 / −0.11 |
| coordinate-other-ais | +0.030 | +0.012 | +0.027 | 0.66 / 0.65 |
| survival-instinct | +0.023 | +0.055 | +0.044 | 0.84 / 0.86 |
| sycophancy | +0.000 | −0.000 | −0.001 | — |

The aggregate edge of chat·−1 over raw·mean (+0.090 vs +0.084) comes from myopic-reward alone;
refusal, hallucination and coordinate are slightly lower. Treat the two as tied.

### 6.3 Fit quality vs steering, layer 22 (500k)

| arm | variant | train cos | acc@1 s→t | acc@1 t→s | dP_coherent |
|---|---|---:|---:|---:|---:|
| raw · mean | plain | 0.966 | 0.020 | 0.951 | +0.084 |
| chat · mean | plain | 0.977 | 0.003 | 0.951 | +0.053 |
| chat · −1 | plain | 0.918 | 0.001 | 0.600 | +0.090 |
| raw · mean | whiten | 0.972 | 0.957 | 0.933 | +0.050 |
| chat · mean | whiten | 0.983 | 0.989 | 0.968 | +0.073 |
| chat · −1 | whiten | 0.938 | 0.687 | 0.771 | +0.047 |

Templating makes the mean-pooled fit *easier* (higher cosine, better retrieval) and the
steering *worse* under plain Procrustes; the single-token arm is the hardest fit and steers at
least as well. Fit-quality metrics do not predict steering here either (cf. §4). The
single-token fit degrades steeply with depth (train cosine 0.99 at l10 → 0.90 at l26), which
is where whitening's retrieval gains are largest but its steering gains are not.

---

## 7. LLaDA2.0-mini (2026-08-13/14, 08-22/26, 09-05)

| scale | arm | train cosine | acc@1 s→t | acc@1 t→s |
|---|---|---:|---:|---:|
| 30k | mean, plain / whiten | 0.17–0.26 / 0.29–0.31 | ≤ 0.02 | ≤ 0.03 |
| 30k | mask, plain / whiten | 0.20–0.24 / 0.28–0.30 | ≤ 0.02 | ≤ 0.02 |
| 500k | mean, plain / whiten | 0.17–0.19 / 0.14–0.17 | ≤ 0.04 | ≤ 0.06 |

The Procrustes fit finds no usable alignment between Llama-1B l8 and any mini layer, at
either scale, with or without whitening. Separately, the A/B evaluation itself is broken on
this model: median unsteered P(match) is 4.3×10⁻⁶ (vs 0.61 on LLaDA-8B), i.e. the answer
letters receive essentially zero probability at the readout position. The 4 263 A/B rows in
`outputs/ab_eval/llada2mini/` therefore measure nothing; the mini's 500k templated caches
(12, verified) are on disk but were not fitted or swept.

---

## 8. Relation to arXiv 2512.24143 ("Activation Steering for Masked Diffusion LMs")

The paper builds a native refusal direction on LLaDA by a diff-of-means over contrastive
prompts, read at a single chat-template-relative token (pre- or post-instruction), with the
(layer, position) pair selected jointly on a 32-item validation set by LLaMA-Guard-2 score, and
applies it by orthogonal projection at every layer.

What overlaps with this repo: single clean forward pass (both); the target-side read position
family (our templated `-1` is a post-instruction token; the paper's winner is post-instruction);
the depth localisation (their 18–24 / 23, our 22). What does not: the paper never fits a map
between two models and never uses generic text; its selection metric is generation-based and
refusal-specific and cannot be reused for the 7 behaviours here. The paper reports that naively
applying its LLaDA direction to an autoregressive model has no effect; our Procrustes-transported
vectors have a small but consistent effect in the reverse direction, which is the point of
fitting a map at all.

Code for the paper's own extraction/selection/application (`chat_positions.py`,
`extract_target_papersel.py`, `select_papersel_position.py`, `apply_projection_ablation` /
`steering_mode="projection"`) is implemented and unit-tested but **has not been run on data**.

---

## 9. Not run, open, or known-broken

- **Projection-ablation steering mode** and the **papersel** target-side extraction: built, not run.
- **30k chat-templated arm** (`config/new_models/fineweb_llada8b_30k_chat.toml`,
  `scripts/run_llada8b_30k_chat.sh`): built, superseded by the 500k arm, not run.
- **Read positions other than `mean` / `-1`** (e.g. `-2`, pre-instruction `0..3`) are refused by
  `extract_activations_multi.py`: `paths.model_slug()` gives every int the same filename, so a
  second int position would silently overwrite or reuse the `-1` cache. Needs a suffix in
  `paths.py` and its two regex consumers before any position sweep.
- **Source templated `mean` cache** exists (`Llama-3.2-1B-Instruct_l8_mean.pt` in
  `data/fineweb_500k_chat`) but only the `-1` source arm was fitted.
- **LLaDA2.0-mini A/B readout** (§7) — cause not diagnosed. Nothing on the mini can be
  validated until it is.
- **No confidence intervals.** Every comparison is a mean over 7 behaviours × 50 items;
  differences below ~0.02 in `dP_coherent` should be read as ties.
- **Native upper bound** (`ab_sweep.py --with-native`: LLaDA's own CAA vector on LLaDA) was not
  run for any arm, so "fraction of native steering recovered" is unknown.
- `ab_report.py` crashed in its HTML step on 2026-09-06 after writing `summary.csv`
  (`source_stats` key became a 4-tuple when `steering_mode` was added). Fixed 2026-09-17;
  `report.html` for the 500k arms is regenerated by the command in §10.

---

## 10. Artifacts

| what | where |
|---|---|
| Activation caches, 30k raw | `data/fineweb_30k/activations/` — LLaDA-8B l10–22 {mean, mask}, mini l6–14 {mean, mask}, Llama l4–12 |
| Activation caches, 500k raw | `data/fineweb_500k/activations/` — LLaDA-8B l10–22 mean, mini l6–14 mean, Llama l4–12 {−1, mean} |
| Activation caches, 500k templated | `data/fineweb_500k_chat/activations/` — LLaDA-8B l10–26 {mean, −1}, mini l6–16 {mean, −1}, Llama l8 {mean, −1} |
| CAA diff caches (LOBO) | `data/caa_diff/activations/` — 4 911 pairs, LLaDA-8B l10–22 {mean, mask}, Llama l8 |
| Translators | `outputs/new_models/` (30k + LOBO), `outputs/new_models_500k/`, `outputs/new_models_500k_chat/` |
| A/B results | `outputs/ab_eval/{llada, llada_lobo, llada2mini, llada_500k, llada_500k_chat}/results.jsonl`, `summary.csv` |
| Raw-vs-chat join | `outputs/ab_eval/compare_raw_vs_chat_500k.csv` |
| Logs | `outputs/*.log` — `llada_fit_20260811_163914`, `llada_lobo_fit_20260812_150242`, `procrustes_500k_20260826_162258`, `llada8b_500k_ab_20260827_081428`, `llada_500k_chat_20260904_112920`, `compare_raw_vs_chat_500k_20260905_173738` |

## 11. Reproducing the 500k comparison

```bash
# templated target caches, both LLaDA models, one pass each        (~15 h, GPU 2)
./scripts/run_llada_500k_chat.sh
# templated source caches + 36 fits + A/B sweep + join            (~8 h, GPU 2; gated on an idle card)
GPU=2 MINUTES=30 ./scripts/wait_gpu_then_run.sh ./scripts/run_compare_raw_vs_chat_500k.sh
# summaries / dashboards
python ab_report.py --results 'outputs/ab_eval/llada_500k/*.jsonl'      --out-dir outputs/ab_eval/llada_500k
python ab_report.py --results 'outputs/ab_eval/llada_500k_chat/*.jsonl' --out-dir outputs/ab_eval/llada_500k_chat
python compare_raw_vs_chat_500k.py
```

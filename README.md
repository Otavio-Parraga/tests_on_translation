# Activation Space Translator

Train a lightweight neural network to translate activation vectors from one LLM's representation space into another's. This enables cross-model transfer of steering vectors, concept probes, and other activation-level representations without needing paired fine-tuning data at the weight level.

## Motivation

Activation engineering techniques — steering vectors, concept probes, representation reading — produce vectors that live in a specific model's internal representation space. These vectors are not directly portable: a steering vector extracted from Llama cannot be applied to Gemma. Typically this means repeating the extraction pipeline for each model.

This project learns a cheap mapping between spaces instead. Given paired activations from two models on the same prompts, a small MLP learns to predict what Gemma's representation of a sentence would look like given only Llama's. Once trained, any source-side activation — including steering vectors computed once — can be projected into the target model's space in a single forward pass.

## Pipeline Overview

```
sample_sentences.py          (optional: generate training sentences from the models themselves)
        ↓
prepare_activations.py       (run both models on all sentences, save activation pairs)
        ↓
train.py                     (train the translator MLP or encoder)
        ↓
evaluate.py                  (top-k retrieval accuracy on held-out activations)
        ↓
translate_steering_vector.py (apply the trained translator to a pre-computed steering vector)
```

---

## Quick Start

### Step 1 — Prepare training sentences

**Option A — Use an existing dataset (MWE, CAA, or TQA)**

```bash
# config/default.toml uses MWE; adjust [dataset] as needed
python prepare_activations.py --config config/default.toml
```

**Option B — Generate sentences directly from the models**

```bash
# Sample N sentences from each model, combine, and save to a JSON file
python sample_sentences.py --n 5000 --config config/default.toml

# Then extract activations using the generated sentences
python prepare_activations.py --config config/generated.toml
```

`sample_sentences.py` loads each model in turn, generates sentences from the BOS token using stochastic decoding (`do_sample=True`), and deduplicates before saving. This covers both models' natural output distributions without requiring an external corpus.

Activations are cached **per model** under the run's `output_dir`, keyed by model name and layer:

```
<output_dir>/activations/<ShortModelName>_l<layer>.pt    # e.g. Llama-3.2-1B-Instruct_l8.pt
```

Each file is a dict with keys `"activations"` `[N, D]`, `"sentences"` (list), and `"n_sentences"`. Keying caches by model means switching one side (e.g. a different target model) reuses the other side's cache instead of re-extracting it, and never overwrites it. (Pre-refactor `source_activations.pt` / `target_activations.pt` files are auto-migrated into this layout on the next `prepare_activations.py` run when they match the sentence set.)

If two GPUs are available, `prepare_activations.py` runs source and target extraction in parallel across `cuda:0` and `cuda:1`. Extraction is resumable: partial batches are saved to a sibling `.batches/` directory and merged on completion, so interrupted runs continue from where they left off.

### Step 2 — Train the translator

```bash
python train.py --config config/default.toml
```

Optional overrides:

```bash
python train.py \
    --config config/default.toml \
    --source-activations /path/to/source_acts.pt \
    --target-activations /path/to/target_acts.pt \
    --output outputs/my_translator.pt
```

The trainer:
- Applies optional L2 normalization to activations (controlled by `normalize_activations`)
- Randomly splits data 80/20 into train and validation sets (seeded for reproducibility)
- Runs a linear LR warmup followed by cosine annealing decay
- Clips gradients to `grad_clip` max norm
- Saves the best checkpoint (by validation loss) to `<output_dir>/best_translator__<source>__<target>__<type>__<loss>.pt`, and the final model to `translator__<source>__<target>__<type>__<loss>.pt`. The experiment-specific name (model pair + translator type + loss) lets MLP/cosine/InfoNCE/SAE runs on the same model pair coexist in one directory without overwriting each other.
- Logs loss curves, learning rate, and top-k retrieval accuracy to TensorBoard after every epoch

Monitor training:

```bash
tensorboard --logdir outputs/tensorboard
```

### Step 3 — Evaluate

```bash
python evaluate.py --config config/default.toml --ks 1 5 10
```

Reports top-k retrieval accuracy (see [Evaluation Metric](#evaluation-metric) below).

### Step 4 — Translate a steering vector

Translate any steering vector from source model space to target model space:

```bash
python translate_steering_vector.py \
    path/to/steering_vectors/meta-llama_Llama-3.2-1B-Instruct/caa/sycophancy/residual/layer_8/sv.pt \
    --checkpoint outputs/best_translator__Llama-3.2-1B-Instruct_l8__gemma-3-1B-it_l8__mlp__cosine.pt
```

Or translate a bare `.pt` tensor without the `activation_engineering` folder structure:

```bash
python translate_vector.py \
    --checkpoint outputs/best_translator__Llama-3.2-1B-Instruct_l8__gemma-3-1B-it_l8__mlp__cosine.pt \
    --input my_vector.pt \
    --output translated_vector.pt
```

---

## Configuration Reference

All settings live in `config/default.toml`. Three pre-built configs are included:

| Config | Description |
|---|---|
| `config/default.toml` | MSE loss, MWE dataset |
| `config/generated.toml` | MSE loss, model-generated sentences |
| `config/combined.toml` | MSE + InfoNCE loss |
| `config/cosine.toml` | Cosine similarity loss |

### `[source_model]` / `[target_model]`

| Field | Description |
|---|---|
| `name` | HuggingFace model ID |
| `layer` | Transformer layer to hook (0-indexed) |
| `module` | Hook point: `"residual"` (post-residual stream) or `"mlp"` |
| `token_position` | Token to extract; `-1` means the last token |

Activations are extracted in `bfloat16` and stored as the hidden state at `token_position` of the specified layer.

### `[dataset]`

| Field | Description |
|---|---|
| `name` | `"MWE"`, `"CAA"`, `"TQA"`, `"GENERATED"`, or `"FINEWEB"` |
| `behaviors` | Behavior filter list; empty or omitted means all |
| `split` | Dataset split: `"generate"`, `"train"`, or `"test"` |
| `data_root` | Root directory of dataset files (or path to the sentences JSON for `"GENERATED"`) |
| `limit` | Max number of sentences to use (optional) |

**Dataset formats:**

- **MWE** — JSONL files under `data_root/evals/`. Each row is expected to have `question`, `answer_matching_behavior`, and `answer_not_matching_behavior` fields. Three sentences are generated per row (question, question+positive, question+negative).
- **CAA** — JSON files under `data_root/CAA_datasets/{split}/{behavior}/`. Same three-sentence structure as MWE.
- **TQA** — CSV pointed to by the `TQA_PATH` environment variable. Expects `Question`, `Best Answer`, and `Incorrect Answers` columns.
- **GENERATED** — A JSON file (list of strings) produced by `sample_sentences.py`. `data_root` is the path to that file.
- **FINEWEB** — Streams `fineweb-edu` (`sample-10BT`) from the network, capped at `limit` sentences (30,000 in every config here, see `FINEWEB_LIMIT` in `acttrans.constants`) and split 80/20 train/test at the document level (seed 42). `data_root` is unused. This is the generic-language dataset behind the cross-model sweep described in [Cross-Model Targets & the FineWeb Sweep](#cross-model-targets--the-fineweb-sweep) below, in place of behavior-specific MWE/CAA sentences.

### `[translator]`

| Field | Description |
|---|---|
| `type` | `"mlp"`, `"encoder"`, `"linear"`, `"sae"`, or `"flow"` |
| `hidden_dims` | List of hidden layer widths (MLP only) |
| `dropout` | Dropout probability applied between layers |
| `activation` | Nonlinearity: `"gelu"`, `"relu"`, or `"silu"` |
| `use_residual` | Add residual connections where dimensions match (MLP only) |
| `d_model` | Internal width of the transformer encoder (encoder only) |
| `nhead` | Attention heads (encoder only) |
| `num_layers` | Encoder stack depth (encoder only) |

**MLP architecture** — stacks `Linear → LayerNorm → Activation → Dropout` blocks, with an optional residual path when input and hidden dimensions match. A final linear layer maps to the target dimension.

**Encoder architecture** — projects the input to `d_model`, passes it through a standard `TransformerEncoder`, then projects to the target dimension.

**Linear translator** — a single (optionally biased) linear layer. Used both as a plain learned baseline and, via `fit_procrustes.py`, as a closed-form orthogonal Procrustes fit (no gradient descent) — see [Cross-Model Targets & the FineWeb Sweep](#cross-model-targets--the-fineweb-sweep).

**SAE / Flow translators** — `sae` is a sparse-autoencoder-style translator (top-k or L1-sparse latent bottleneck, `latent_dim`/`k`/`l1_coeff`); `flow` is an invertible coupling-block normalizing flow (`num_blocks`/`coupling_hidden`/`mixing`). Both are experimental alternatives to the MLP, configured under `config/loss_combos/` and `config/fineweb/`; see `src/acttrans/models/translator.py` for the full field list of each.

### `[training]`

| Field | Description |
|---|---|
| `epochs` | Total training epochs |
| `batch_size` | Mini-batch size |
| `lr` | Peak learning rate (AdamW) |
| `weight_decay` | L2 regularisation |
| `train_ratio` | Fraction of data used for training (rest is validation) |
| `seed` | Random seed for the train/val split |
| `output_dir` | Directory for checkpoints and activation files |
| `normalize_activations` | L2-normalize source and target activations before training |
| `losses` | List of loss components: `"mse"`, `"cosine"`, `"info_nce"` |
| `loss_weights` | Scalar weight for each loss component (same order as `losses`) |
| `temperature` | Softmax temperature for InfoNCE (only used when `"info_nce"` is in `losses`) |
| `grad_clip` | Max gradient norm; `0.0` disables clipping |
| `lr_warmup_epochs` | Linear warmup epochs before cosine decay begins; `0` disables warmup |

**Loss functions:**

| Name | Formula | When to use |
|---|---|---|
| `mse` | Mean squared error | Default; directly minimises distance in activation space |
| `cosine` | `1 - cos(pred, target)` | When direction matters more than magnitude |
| `info_nce` | Symmetric contrastive loss over normalised embeddings | Improves retrieval accuracy; combine with `mse` at a lower weight |

---

## Evaluation Metric

**Top-k Retrieval Accuracy** measures how well the translator preserves identity across the full validation set, not just closeness in absolute distance.

For each source activation in the validation set, the translator produces a predicted target. That prediction is ranked by cosine similarity against every target activation in the validation set. Top-k accuracy is the fraction of examples where the correct target appears in the k nearest neighbours.

- **Top-1** — strict: the single closest target must be the correct one
- **Top-5 / Top-10** — lenient: correct target appears anywhere in the k closest

Random baseline scores `k / N`, so for 200 validation samples Top-1 random chance is 0.5%. Retrieval is computed bidirectionally (src→tgt and tgt→src) and logged to TensorBoard each epoch.

---

## Translating Steering Vectors

`translate_steering_vector.py` is designed for integration with the `activation_engineering` repository. It expects a source `sv.pt` located inside a folder tree of the form:

```
steering_vectors/{model}/{method}/{behavior}/{module}/layer_{idx}/sv.pt
```

It outputs the translated vector at:

```
steering_vectors/{target_model}/{method}/{behavior}/{module}/layer_{target_layer}/sv.pt
steering_vectors/{target_model}/{method}/{behavior}/{module}/layer_{target_layer}/sv.json
```

By default the translated vector is rescaled to match the source vector's original L2 norm (`--no-restore-norm` disables this). The metadata JSON is initialised from the source `sv.json` if present, then updated with target model fields and a provenance trail.

```bash
python translate_steering_vector.py \
    path/to/sv.pt \
    --checkpoint outputs/best_translator__Llama-3.2-1B-Instruct_l8__gemma-3-1B-it_l8__mlp__cosine.pt \
    --output-dir /path/to/output/root
```

For a bare tensor without the folder structure:

```bash
python translate_vector.py \
    --checkpoint outputs/best_translator__Llama-3.2-1B-Instruct_l8__gemma-3-1B-it_l8__mlp__cosine.pt \
    --input my_vector.pt \
    --output translated.pt
```

The input can be a 1-D vector `[D]` or a batch `[N, D]`; a 1-D input is unsqueezed automatically.

---

## Discovery Methods (CAA, RepE, GCAV, BiPO)

`{method}` in the paths above is a real dimension, not a constant. Four of
`activation_engineering`'s steering-vector discovery methods are supported, listed in
`acttrans.constants.METHODS`:

| Method | How the direction is found | Saved shape | Norm |
|---|---|---|---|
| **CAA** | mean of per-pair (positive − negative) activation differences | `[1, D]` | its own, behavior-dependent (~0.2–5) |
| **RepE** | 1st principal component of sign-alternated, per-pair-normalized differences (LAT) | `[1, D]` | 1 by construction |
| **GCAV** | normal of a logistic-regression boundary separating positive from negative activations | `[D]` | 1 (explicitly normalized) |
| **BiPO** | one vector learned by gradient descent on a bidirectional preference (DPO-style) loss over (question, target, opposite) triplets | `[D]` | its own, behavior-dependent |

All four are a *single flat direction in the residual stream* per
`(model, behavior, layer)`, built from the same CAA contrastive dataset, and applied
with an additive hook — which is exactly why they share this tooling unchanged.
**How the direction is found is irrelevant once it is a vector on disk**: BiPO is
trained by gradient descent rather than derived in closed form, and still needs no
special handling. `load_sv` squeezes dim 0, so the `[1, D]` / `[D]` split needs no
special casing either.

> **BiPO caveat — ignore `current_direction` in its `sv.json`.** That field is a
> transient training variable (a fresh `d ~ U{-1,+1}` sampled *every minibatch* to
> toggle the perturbation sign); what got serialized is just the last draw of the last
> epoch, and it is ~50/50 across behaviors and layers. Using it to flip signs would
> **randomize** them. The learned vector's orientation is already consistent — the loss
> makes `+v` steer toward the target behavior — confirmed empirically: `cos(BiPO, CAA)`
> on the same `(model, behavior, layer)` is positive for 74% of 224 pairs, and is
> 75%/72% for `current_direction=+1/−1`, i.e. the field carries no sign information.

```bash
# geometric + decomposition comparison across methods (cheap, CPU)
python compare_translated_and_original.py all --methods CAA RepE GCAV BiPO

# A/B steering sweep across methods (one row per method/scope/translator/behavior/coeff)
python ab_sweep.py --methods CAA RepE GCAV BiPO --with-native

# per-method report (single method per page), then the cross-method view
python ab_report.py   --results outputs/ab_eval/methods/results.jsonl --methods RepE
python method_report.py --results outputs/ab_eval/methods/results.jsonl
```

### The one thing that is not comparable across methods: scale

CAA and BiPO vectors carry their own magnitude while RepE and GCAV are unit vectors,
so **the same `coefficient` is a different physical perturbation per method.** Anything
that compares methods must correct for this; anything that compares translators
*within* one method need not. `acttrans.constants.UNIT_NORM_METHODS` records which side
each method falls on.

- **Cosine-based analysis** (`compare_translated_and_original.py` → `geometric`,
  `decomposition`) is scale-free, so it is method-comparable as-is. Every output
  table gains a leading `method` column.
- **A/B steering** is not. `method_report.py` therefore reads each run on
  `dose = coefficient × sv_norm` and scores it by
  `dP_peak = maxᶜ[P(match|+c) − P(match|−c)]` — the best swing at *any* dose, with
  the dose reported alongside. Because a collapsed model has `P(match) → 0` on both
  sides, collapsed doses contribute ~0 swing and cannot win the peak search, so no
  arbitrary coherence cutoff is needed.
- `ab_report.py`, `ab_dashboard.py` and `ab_pivot_dashboard.py` share one raw
  coefficient axis, so they are **single-method by construction** and take a
  `--method(s)` flag. `ab_report.py` warns if handed a mixed-method file.

Result rows written before methods were a dimension carry no `method` field; every
reader backfills a missing method to `CAA`, so old JSONL/CSV files still resume,
de-duplicate and report correctly.

---

## Cross-Model Targets & the FineWeb Sweep

The source model is fixed throughout this project — `meta-llama/Llama-3.2-1B-Instruct`
— but the target side has grown from one same-family model into a small matrix of
increasingly different architectures:

| Target | Relationship to source | Notes |
|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | same family | the original experiment; 28 layers, hidden_size 3072 |
| `Qwen/Qwen2.5-0.5B-Instruct` | cross-family | 24 layers, hidden_size 896 |
| `google/gemma-3-1B-it` | cross-family | 26 layers, hidden_size 1152; very large residual-activation scale (see the config's own note) |
| `microsoft/Phi-tiny-MoE-instruct` | cross-architecture | sparse Mixture-of-Experts (16 experts, 2 active/token, 3.8B total / 1.1B active params) — the first non-dense architecture tested here |

Config files for the three newer targets live in `config/new_models/`
(`fineweb_qwen0.5b.toml`, `fineweb_gemma1b.toml`, `fineweb_phitinymoe.toml`); the
shared generic-language dataset they and the original Gemma experiment train on is
`config/fineweb.toml`, with loss/pooling variants under `config/fineweb/` generated
by `python -m acttrans.config_gen.fineweb` and a hand-maintained Procrustes config,
`config/fineweb/linear_procrustes_raw.toml`. The generated variants under
`config/fineweb/` and `config/loss_combos/` are **git-ignored** (regenerable output);
only the hand-maintained `linear_procrustes_raw.toml` is tracked. Regenerate with:

```bash
conda run -n acteng python -m acttrans.config_gen.fineweb        # -> config/fineweb/
conda run -n acteng python -m acttrans.config_gen.loss_combos    # -> config/loss_combos/
```

These configs use `[dataset] name = "FINEWEB"` instead of MWE/CAA/GENERATED — see
above — because fitting a translator that has to hold up across several target
architectures benefits from generic-language coverage rather than behavior-specific
sentences.

### Workflow

```bash
# 1. Extract ONE model's activations across a layer sweep, reusing the cached
#    FineWeb sentences (no network access once the sentence cache exists):
conda run -n acteng python extract_activations_sweep.py \
    --config config/new_models/fineweb_qwen0.5b.toml --layers 8 10 12 14 16

# 2. Before training anything, check whether the two spaces are even linearly
#    alignable (CKA, mutual k-NN, mean-collapse ratio):
conda run -n acteng python diagnose_alignment.py --config config/new_models/fineweb_qwen0.5b.toml

# 3a. Fit the closed-form orthogonal Procrustes baseline (translator.type = "linear";
#     every config in config/new_models/ uses this) ...
conda run -n acteng python fit_procrustes.py --config config/fineweb/linear_procrustes_raw.toml
# 3b. ... or train a learned translator (mlp/encoder/sae/flow) the usual way:
conda run -n acteng python train.py --config config/fineweb/mlp_mse_last.toml

# 4. Run the A/B steering sweep and build the dashboards (see "Discovery Methods"
#    above for the --methods axis; --with-native also evaluates each target
#    model's OWN from-scratch CAA vector as a translation-quality ceiling):
conda run -n acteng python ab_sweep.py --with-native
conda run -n acteng python ab_dashboard.py        # outputs/ab_eval/dashboard.html — every sweep, one page
conda run -n acteng python ab_pivot_dashboard.py  # outputs/ab_eval/pivot_dashboard.html — freely pivotable
```

`extract_activations_sweep.py` only ever extracts the config's `[target_model]` side,
reusing the persisted `sentences.json` so every layer's cache stays row-aligned with
the source-model cache already produced by `prepare_activations.py`/`fineweb.toml`
(same 30k FineWeb sentences, same order). Each `(model, layer)` cache is independent
and keyed by model name, so `fit_procrustes.py`/`train.py` transparently reuse
whichever caches already exist for the pair a config names.

### Phi-tiny-MoE needs extra setup

Unlike Llama/Qwen/Gemma, `microsoft/Phi-tiny-MoE-instruct` required real code and
environment changes, not just a new config:

- **Code** (`src/acttrans/utils/hf.py`): `load_model_and_tokenizer` always passes
  `trust_remote_code=True` — required because Phi ships custom modeling code
  (`PhiMoEForCausalLM`); it's a no-op for the other models. The same file also
  monkey-patches `DynamicCache.get_usable_length` back in, since newer `transformers`
  dropped that method but Phi's custom code still calls it.
- **Environment** (the `acteng` conda env; not tracked by git): a real `einops`
  install and a minimal local `flash_attn` stub package, both needed just to *import*
  Phi's modeling code — attention still runs via the sdpa/eager fallback, since no
  flash-attn kernels exist for the Pascal-generation GPUs this project runs on.

---

## Generating Training Data from the Models

When no external dataset is available, `sample_sentences.py` generates a training corpus by sampling directly from both models:

```bash
python sample_sentences.py \
    --config config/default.toml \
    --n 5000 \
    --output outputs/generated_sentences.json \
    --temperature 1.0 \
    --max_new_tokens 64
```

Each model is loaded once, generates `--n` sentences starting from the BOS token using `do_sample=True`, then is unloaded before the next model loads. The output is a deduplicated JSON list of all sentences from both models combined.

The resulting file is consumed by `prepare_activations.py` via `config/generated.toml`, where `dataset.name = "GENERATED"` and `data_root` points to the JSON file.

**Trade-offs vs. a curated dataset:**  
Stochastic decoding from BOS covers both models' natural output distributions and requires no external data. Coverage depends on generation diversity — higher temperature and larger `--n` improve breadth. The sentences will reflect each model's stylistic biases (fluent, coherent, assistant-register), which suits use cases where the downstream inputs are also LLM-generated.

---

## Installation

The library code lives in the installable `acttrans` package. The runtime
environment is the `acteng` conda env (which already ships torch, transformers,
etc.), so install without touching its dependencies:

```bash
conda run -n acteng pip install -e . --no-deps
```

Every entry point is a script at the repo root; run them from the repo root
(paths in the configs are relative to it). Convenience shell wrappers live in
`scripts/` (each `cd`s to the repo root itself, so they work from anywhere).

Run the test suite with:

```bash
conda run -n acteng pip install -e '.[test]' --no-deps   # once, adds pytest
conda run -n acteng python -m pytest
```

## Project Structure

```
├── pyproject.toml               Installable package definition (pip install -e . --no-deps)
├── prepare_activations.py       Extract paired activations from source and target models
├── extract_activations_sweep.py Extract ONE model's activations across a layer sweep (FineWeb, cross-model targets)
├── sample_sentences.py          Generate training sentences via stochastic decoding
├── train.py                     Train the translator
├── evaluate.py                  Evaluate top-k retrieval accuracy on saved activations
├── fit_procrustes.py            Closed-form orthogonal-Procrustes linear baseline
├── diagnose_alignment.py        Stage-0 alignment diagnostics (CKA, mutual k-NN, collapse)
├── translate_vector.py          Translate a bare .pt tensor
├── translate_steering_vector.py Translate a steering vector (activation_engineering layout)
├── ab_comparison.py             A/B eval: original vs translated SV, one behavior
├── ab_sweep.py                  Full A/B steering sweep over all translators x methods (resumable)
├── ab_report.py                 Tables + HTML report from ab_sweep results (one method per report)
├── method_report.py             Cross-method report (CAA/RepE/GCAV/BiPO) on the dose axis
├── ab_dashboard.py              Consolidated multi-run HTML dashboard, all sweeps (one method per page)
├── ab_pivot_dashboard.py        Freely pivotable HTML dashboard (one method per page)
├── compare_translated_and_original.py  Geometric/decomposition SV comparison (method-aware)
├── config/
│   ├── default.toml             MSE loss, MWE dataset
│   ├── generated.toml           MSE loss, model-generated sentences
│   ├── combined.toml            MSE + InfoNCE
│   ├── cosine.toml              Cosine loss
│   ├── fineweb.toml              FineWeb-edu dataset, generic-language translator training (Llama->Gemma)
│   ├── fineweb/                  GENERATED arch x loss x pooling configs (git-ignored) + hand-written linear_procrustes_raw.toml (tracked)
│   ├── loss_combos/              GENERATED compound-loss configs (git-ignored)
│   ├── layer_sweep/              Llama 1B<->3B cross-layer transfer sweep configs
│   └── new_models/               Cross-family/cross-architecture targets: Qwen2.5-0.5B, gemma-3-1B, Phi-tiny-MoE
├── scripts/                     Convenience shell wrappers (sweeps, extraction, translation)
├── tests/                       pytest suite (paths, split, losses, metrics, translator, methods)
├── src/acttrans/                The installable package
│   ├── constants.py             Fixed experiment grid (model pair, layer, behaviors, METHODS)
│   ├── data/
│   │   ├── dataset.py           Activation extraction (batch resume) + per-dataset sentence loaders
│   │   └── split.py             Seeded train/val split + activation preprocessing (shared)
│   ├── models/
│   │   ├── translator.py        MLP/Encoder/SAE/Flow/Linear translators + Procrustes fit
│   │   └── transport.py         TranslatorRunner: in-memory direction transport
│   ├── training/
│   │   ├── trainer.py           Training loop with TensorBoard logging
│   │   └── losses.py            Coordinate/relational losses (mse, cosine, info_nce, vsp)
│   ├── evaluation/
│   │   ├── evaluator.py         Checkpoint evaluation entry point
│   │   ├── metrics.py           Top-k retrieval accuracy
│   │   ├── ab_eval.py           Steering hook + closed-ended A/B evaluation
│   │   ├── method_compare.py    Dose-axis metrics for cross-method comparison
│   │   ├── response_metrics.py  Shared P-curve stats (Pearson/Spearman/effect size) for report + dashboards
│   │   └── results_io.py        Shared A/B result loading (JSONL/CSV) + translator-name parsing
│   ├── config_gen/             Training-sweep config generators (python -m acttrans.config_gen.{fineweb,loss_combos})
│   │   ├── common.py            Shared model/dataset/translator TOML blocks (was config/sweep_common.py)
│   │   ├── fineweb.py           arch x loss x pooling sweep -> config/fineweb/
│   │   └── loss_combos.py       compound-loss sweep -> config/loss_combos/
│   ├── comparison/              Translated-vs-native SV analyses (geometric, decomposition)
│   └── utils/
│       ├── paths.py             Path/slug conventions (per-model / per-experiment / SV trees)
│       ├── config.py            TOML loading + activation-cache resolution
│       ├── checkpoints.py       Translator checkpoint discovery/metadata parsing
│       └── hf.py                HF model/tokenizer loading + layer lookup (+ Phi-tiny-MoE trust_remote_code/cache shim)
└── outputs/<sentence_set>/      One directory per sentence set (holds sentences.json)
    ├── activations/
    │   ├── <SrcModel>_l<layer>.pt   Per-model activation cache (shared across experiments)
    │   └── <TgtModel>_l<layer>.pt
    ├── best_translator__<src>__<tgt>__<type>__<loss>.pt   Best checkpoint (per experiment)
    ├── translator__<src>__<tgt>__<type>__<loss>.pt        Final checkpoint
    └── tensorboard/             TensorBoard run directories
```

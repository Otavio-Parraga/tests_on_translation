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
| `name` | `"MWE"`, `"CAA"`, `"TQA"`, or `"GENERATED"` |
| `behaviors` | Behavior filter list; empty or omitted means all |
| `split` | Dataset split: `"generate"`, `"train"`, or `"test"` |
| `data_root` | Root directory of dataset files (or path to the sentences JSON for `"GENERATED"`) |
| `limit` | Max number of sentences to use (optional) |

**Dataset formats:**

- **MWE** — JSONL files under `data_root/evals/`. Each row is expected to have `question`, `answer_matching_behavior`, and `answer_not_matching_behavior` fields. Three sentences are generated per row (question, question+positive, question+negative).
- **CAA** — JSON files under `data_root/CAA_datasets/{split}/{behavior}/`. Same three-sentence structure as MWE.
- **TQA** — CSV pointed to by the `TQA_PATH` environment variable. Expects `Question`, `Best Answer`, and `Incorrect Answers` columns.
- **GENERATED** — A JSON file (list of strings) produced by `sample_sentences.py`. `data_root` is the path to that file.

### `[translator]`

| Field | Description |
|---|---|
| `type` | `"mlp"` or `"encoder"` |
| `hidden_dims` | List of hidden layer widths (MLP only) |
| `dropout` | Dropout probability applied between layers |
| `activation` | Nonlinearity: `"gelu"`, `"relu"`, or `"silu"` |
| `use_residual` | Add residual connections where dimensions match (MLP only) |
| `d_model` | Internal width of the transformer encoder (encoder only) |
| `nhead` | Attention heads (encoder only) |
| `num_layers` | Encoder stack depth (encoder only) |

**MLP architecture** — stacks `Linear → LayerNorm → Activation → Dropout` blocks, with an optional residual path when input and hidden dimensions match. A final linear layer maps to the target dimension.

**Encoder architecture** — projects the input to `d_model`, passes it through a standard `TransformerEncoder`, then projects to the target dimension.

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

## Project Structure

```
├── prepare_activations.py       Extract paired activations from source and target models
├── sample_sentences.py          Generate training sentences via stochastic decoding
├── train.py                     Train the translator
├── evaluate.py                  Evaluate top-k retrieval accuracy on saved activations
├── translate_vector.py          Translate a bare .pt tensor
├── translate_steering_vector.py Translate a steering vector (activation_engineering layout)
├── config/
│   ├── default.toml             MSE loss, MWE dataset
│   ├── generated.toml           MSE loss, model-generated sentences
│   ├── combined.toml            MSE + InfoNCE
│   └── cosine.toml              Cosine loss
├── src/
│   ├── data/dataset.py          Activation extraction with batch-level resume
│   ├── models/translator.py     MLPTranslator, EncoderTranslator, SparseAutoencoderTranslator
│   ├── training/trainer.py      Training loop with TensorBoard logging
│   ├── utils/paths.py           Output path conventions (per-model / per-experiment)
│   └── evaluation/
│       ├── evaluator.py         Checkpoint evaluation entry point
│       └── metrics.py           Top-k retrieval accuracy
└── outputs/<sentence_set>/      One directory per sentence set (holds sentences.json)
    ├── activations/
    │   ├── <SrcModel>_l<layer>.pt   Per-model activation cache (shared across experiments)
    │   └── <TgtModel>_l<layer>.pt
    ├── best_translator__<src>__<tgt>__<type>__<loss>.pt   Best checkpoint (per experiment)
    ├── translator__<src>__<tgt>__<type>__<loss>.pt        Final checkpoint
    └── tensorboard/             TensorBoard run directories
```

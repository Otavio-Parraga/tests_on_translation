#!/usr/bin/env bash
# Shared environment for the LLaDA extraction launchers (sourced, not executed).
#
# Why this file exists instead of the `RUN="conda run -n acteng python"` line every
# other script in scripts/ uses: on THIS machine that env name does not resolve.
# There is no ~/.condarc, and conda's envs_dirs is only
#   /home/parraga/miniconda3/envs  +  /home/parraga/.conda/envs
# while the environment actually lives at /mnt/C-SSD/parraga/conda_envs/act_eng.
# So `conda run -n act_eng ...` fails with EnvironmentLocationNotFound and the
# path form (`-p`) is the only one that works. Override CONDA_ENV_PATH to relocate.
#
# --no-capture-output makes conda stream stdout/stderr live instead of buffering
# until exit — essential when the run is detached and followed with `tail -f`.

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/C-SSD/parraga/conda_envs/act_eng}"
GPU="${GPU:-2}"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

# The weights cache. .env also carries this (extract_activations_sweep.py calls
# load_dotenv()), but export it here too so the launchers work even if .env is
# missing. python-dotenv does not override already-set variables, so whichever
# value is exported here wins — which is also why .env's own CUDA_VISIBLE_DEVICES
# (if it ever gains one) can never steal the GPU pinned above.
export HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/E-SSD/model_cache/hf}"

RUN="conda run -p $CONDA_ENV_PATH --no-capture-output python"

# --- preflight: fail fast, before loading tens of GB of weights ---------------
preflight() {
  local rc=0
  if [[ ! -x "$CONDA_ENV_PATH/bin/python" ]]; then
    echo "!! conda env python not found: $CONDA_ENV_PATH/bin/python" >&2
    echo "   (set CONDA_ENV_PATH=/path/to/env to override)" >&2
    rc=1
  fi
  if [[ ! -d "$HF_CACHE_DIR" ]]; then
    echo "!! HF_CACHE_DIR does not exist: $HF_CACHE_DIR" >&2
    echo "   Without it HF falls back to ~/.cache/huggingface and re-downloads ~32GB." >&2
    rc=1
  fi
  # sentences.json is the ground truth for the 30k FineWeb corpus. If it were
  # missing, extract_activations_sweep.py would silently re-stream FineWeb from
  # the network and produce activations that are NOT row-aligned with the
  # existing Llama source caches — the one failure mode that would quietly
  # invalidate every downstream Procrustes fit. Refuse to start instead.
  if [[ ! -f data/fineweb/sentences.json ]]; then
    echo "!! data/fineweb/sentences.json is missing." >&2
    echo "   Refusing to run: extraction would re-stream FineWeb and break row alignment." >&2
    rc=1
  fi
  if (( rc != 0 )); then
    echo "preflight FAILED — nothing was run." >&2
    return 1
  fi

  # Prove the env python actually RUNS (not merely that the file exists) and that
  # the corpus it will read is the expected 30k FineWeb set — one cheap check that
  # covers both. Deliberately imports only json: no torch, so no CUDA context is
  # created and no GPU memory is touched by the preflight itself.
  local probe
  probe=$("$CONDA_ENV_PATH/bin/python" -c \
    'import json,sys,acttrans; print(sys.executable, len(json.load(open("data/fineweb/sentences.json"))))' 2>&1) || {
      echo "!! env python failed to run, 'import acttrans' failed, or sentences.json is unreadable:" >&2
      echo "   $probe" >&2
      echo "preflight FAILED — nothing was run." >&2
      return 1
    }
  local interp="${probe%% *}" n_sent="${probe##* }"
  if [[ "$n_sent" != "30000" ]]; then
    echo "!! sentences.json holds $n_sent sentences, expected 30000 (FINEWEB_LIMIT)." >&2
    echo "   Refusing to run: target caches would not align with the Llama source caches." >&2
    echo "preflight FAILED — nothing was run." >&2
    return 1
  fi

  echo "preflight OK  |  interpreter=$interp"
  echo "              |  HF_CACHE_DIR=$HF_CACHE_DIR"
  echo "              |  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (-> physical GPU $GPU)"
  echo "              |  corpus=data/fineweb/sentences.json ($n_sent sentences)"
}

"""Training-sweep config generators.

The FineWeb and compound-loss sweeps are emitted from code so their config
matrices can never drift apart: shared model/dataset/translator TOML blocks live
in ``common.py``; each generator is a module you run with ``python -m``:

    conda run -n acteng python -m acttrans.config_gen.fineweb       # -> config/fineweb/
    conda run -n acteng python -m acttrans.config_gen.loss_combos   # -> config/loss_combos/

Run from the repo root: the output directories are resolved relative to the
working directory (``config/…``), matching every other tool in this repo.
"""

# ESMFold2-MLX

ESMFold2 (everything after ESMC-6B) in MLX: 1.58-1.81x faster than torch on Apple-silicon MPS at 247-499 tokens, with lower
peak memory, matching torch on MPS within its own run-to-run spread. Results, decisions and how to reproduce: `HANDOFF.md`.

- Python: `/opt/homebrew/Caskroom/miniforge/base/envs/esmfold2-binder-design/bin/python` (never `conda run`; install nothing into it).
- Convert weights once: `PYTHONPATH=. $PY scripts/convert_weights.py`. Tests: `PYTHONPATH=. $PY -m pytest tests -q` (one GPU run at a time; the session holds `gpu_lock()`).
- Fold: `bin/esmfold2-mlx pred -i job.json -o out/` (job format and options in `esmfold2_mlx/cli.py` and `HANDOFF.md`).

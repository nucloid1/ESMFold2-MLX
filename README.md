# ESMFold2-MLX

ESMFold2 (everything except ESMC-6B) in MLX/Metal. Design: `docs/superpowers/specs/2026-10-29-esmfold2-mlx-design.md`; plan: `docs/superpowers/plans/2026-10-29-esmfold2-mlx.md`.

- Python: `/opt/homebrew/Caskroom/miniforge/base/envs/esmfold2-binder-design/bin/python` (never `conda run`; install nothing into it).
- Convert weights once: `PYTHONPATH=. $PY scripts/convert_weights.py`. Run tests: `PYTHONPATH=. $PY -m pytest tests -q` (one GPU run at a time; the session holds `gpu_lock()`).
- CLI (Task 12): `bin/esmfold2-mlx pred -i job.json -o out/`.

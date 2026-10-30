# Test data

- `cases.json`: copied on 2026-10-29 from `/Users/ethanputnam/Bindcraft-Hunter/docs/superpowers/spikes/spike4b-fast-vs-plain/cases.json` (Spike 4b of BindCraft-Hunter): real target and binder sequences of PDB complexes (`TNF_*`, `IL17_*`, `VEGF_*`) with token counts. This project reads only the sequences.
- `tnfa_AC.a3m`: copied on 2026-10-29 from `/Users/ethanputnam/Bindcraft-Hunter/docs/superpowers/pilots/tnfa-ab/tnfa_AC.a3m`: a real 2225-row alignment whose query is the 152-residue TNF-alpha crystal chain (1TNF chains A/C, mature 6-157; built by that pilot's `prep_msa.py`). `scripts/cases.py` crops it (columns, rows) into `refs/_msa/` with the same column logic as the pilot's `crop_a3m.py`.

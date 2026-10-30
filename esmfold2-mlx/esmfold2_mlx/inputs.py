"""Job files for the CLI: protein complexes only.

    {"jobs": [{"name": "c1", "chains": [{"id": "A", "sequence": "...", "msa": "path.a3m" | null},
                                        {"id": ["B", "C"], "sequence": "..."}]}]}

Only plain amino-acid sequences are validated by this port (spec non-goals), so ligands, DNA/RNA,
modifications and anything else are rejected with a message naming the chain, never folded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")
_NOT_PROTEIN_KEYS = ("type", "smiles", "ccd", "modifications", "dna", "rna", "ligand")


@dataclass(frozen=True)
class Chain:
    ids: tuple[str, ...]
    sequence: str
    msa: str | None = None


@dataclass(frozen=True)
class Job:
    name: str
    chains: tuple[Chain, ...]

    def tokens(self) -> int:
        return sum(len(c.sequence) * len(c.ids) for c in self.chains)

    def states_key(self) -> str:
        """Cache key of the ESMC hidden states: they depend on the chains and sequences only (not the
        job's name, nor its MSAs)."""
        import hashlib

        payload = json.dumps([[list(c.ids), c.sequence] for c in self.chains])
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def to_spec(self):
        """esm ``StructurePredictionInput`` (parent process only: imports esm)."""
        from esm.models.esmfold2.types import MSA, ProteinInput, StructurePredictionInput

        return StructurePredictionInput(sequences=[
            ProteinInput(id=list(c.ids) if len(c.ids) > 1 else c.ids[0], sequence=c.sequence,
                         msa=MSA.from_a3m(c.msa) if c.msa else None) for c in self.chains])


def _chain(raw: dict, job: str) -> Chain:
    ids = raw.get("id")
    ids = tuple(ids) if isinstance(ids, list) else (ids,)
    label = ",".join(str(i) for i in ids)
    if not all(isinstance(i, str) and i for i in ids):
        raise ValueError(f"job {job!r}: chain ids must be non-empty strings, got {raw.get('id')!r}")
    for k in _NOT_PROTEIN_KEYS:
        if k in raw and not (k == "type" and raw[k] == "protein"):
            raise ValueError(f"job {job!r}, chain {label}: {k}={raw[k]!r} is not supported; protein complexes only")
    sequence = raw.get("sequence", "")
    if not sequence:
        raise ValueError(f"job {job!r}, chain {label}: the sequence is empty")
    bad = sorted(set(sequence) - AMINO_ACIDS)
    if bad:
        raise ValueError(f"job {job!r}, chain {label}: characters {bad} are not amino acids; protein complexes only")
    msa = raw.get("msa")
    if msa is not None and not os.path.isfile(msa):
        raise ValueError(f"job {job!r}, chain {label}: no such MSA file {msa!r}")
    return Chain(ids, sequence, msa)


def parse_jobs(path: str) -> list[Job]:
    with open(path) as handle:
        raw = json.load(handle)
    jobs, names = [], set()
    for entry in raw.get("jobs", []):
        name = entry.get("name")
        if not name:
            raise ValueError("every job needs a name")
        if name in names:
            raise ValueError(f"duplicate job name {name!r}")
        names.add(name)
        chains = tuple(_chain(c, name) for c in entry.get("chains", []))
        if not chains:
            raise ValueError(f"job {name!r} has no chains")
        seen = set()
        for c in chains:
            for i in c.ids:
                if i in seen:
                    raise ValueError(f"job {name!r}: duplicate chain id {i!r}")
                seen.add(i)
        jobs.append(Job(name, chains))
    if not jobs:
        raise ValueError(f"{path} contains no jobs")
    return jobs

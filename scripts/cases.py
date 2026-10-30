"""Test cases: protein complexes built from real sequences in data/cases.json.

A chain is ``(ids, sequence, a3m_path | None)``; ``ids`` is one chain id or a tuple of ids for a
homo-multimer. More cases are added by the tasks that need them.
"""

import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_RAW = json.loads((REPO / "data" / "cases.json").read_text())["cases"]
_VEGF = _RAW["VEGF_9KKU"]
_M49 = _VEGF["binder_seq"]  # 43 residues


@dataclass(frozen=True)
class Case:
    name: str
    chains: tuple

    def tokens(self) -> int:
        return sum(len(seq) * (len(ids) if isinstance(ids, tuple) else 1) for ids, seq, _ in self.chains)


CASES = {
    "m49": Case("m49", (("A", _M49, None),)),  # 43 tokens
    "vegf1": Case("vegf1", (("A", _M49, None), ("B", _VEGF["target_seq"], None))),  # 145 tokens
}

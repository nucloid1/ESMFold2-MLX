"""Test cases: protein complexes built from real sequences (data/cases.json) and a real alignment
(data/tnfa_AC.a3m, query = the 152-residue TNF-alpha crystal chain).

A chain is ``(ids, sequence, a3m_path | None)``; ``ids`` is one chain id or a tuple of ids for a
homo-multimer. Cropped alignments are written to refs/_msa/ (gitignored, regenerated on import).
"""

import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_RAW = json.loads((REPO / "data" / "cases.json").read_text())["cases"]
_VEGF = _RAW["VEGF_9KKU"]
_M49 = _VEGF["binder_seq"]  # 43 residues
_MSA_DIR = REPO / "refs" / "_msa"


def _read_a3m(path: Path) -> list[tuple[str, str]]:
    records, header, seq = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header, seq = line, []
        elif line.strip():
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def crop_row(seq: str, start: int, end: int) -> str:
    """Keep the characters of an a3m row lying in query match columns [start, end) (BindCraft-Hunter
    pilots/tnfa-ab/crop_a3m.py): uppercase and '-' consume a match column, lowercase insertions belong
    to the gap before the next column."""
    out, col = [], 0
    for ch in seq:
        if ch.islower():
            if start < col <= end:
                out.append(ch)
        else:
            if start <= col < end:
                out.append(ch)
            col += 1
    return "".join(out)


_TNF_RECORDS = _read_a3m(REPO / "data" / "tnfa_AC.a3m")
_TNF = _TNF_RECORDS[0][1]  # 152 residues


def _write_a3m(name: str, records: list[tuple[str, str]]) -> str:
    _MSA_DIR.mkdir(parents=True, exist_ok=True)
    path = _MSA_DIR / f"{name}.a3m"
    text = "".join(f"{h}\n{s}\n" for h, s in records)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return str(path)


def tnf_a3m(cols: int | None = None, rows: int | None = None) -> str:
    recs = _TNF_RECORDS[:rows] if rows else _TNF_RECORDS
    if cols is not None:
        recs = [(h, crop_row(s, 0, cols)) for h, s in recs]
    return _write_a3m(f"tnf_c{cols or 'all'}_r{rows or 'all'}", recs)


@dataclass(frozen=True)
class Case:
    name: str
    chains: tuple

    def tokens(self) -> int:
        return sum(len(seq) * (len(ids) if isinstance(ids, tuple) else 1) for ids, seq, _ in self.chains)


def _complex(name: str, spec: dict, target_msa: str | None = None, target: str | None = None, copies: int | None = None) -> Case:
    ids = tuple(chr(ord("B") + i) for i in range(copies or spec["target_copies"]))
    return Case(name, (("A", spec["binder_seq"], None), (ids if len(ids) > 1 else ids[0], target or spec["target_seq"], target_msa)))


CASES = {
    # small parity cases (CPU references)
    "m49": Case("m49", (("A", _M49, None),)),  # 43 tokens, single sequence
    "m49_q": Case("m49_q", (("A", _M49, _write_a3m("m49_query_only", [(">query", _M49)])),)),  # 43, a one-row a3m
    "vegf1": Case("vegf1", (("A", _M49, None), ("B", _VEGF["target_seq"], None))),  # 145
    "tnf60_msa": Case("tnf60_msa", (("A", _M49, None), ("B", _TNF[:60], tnf_a3m(cols=60)))),  # 103, 2225 rows, binder without MSA
    "tnf60_shallow": Case("tnf60_shallow", (("A", _M49, None), ("B", _TNF[:60], tnf_a3m(cols=60, rows=200)))),  # 103, 200 rows
    # speed and memory cases (MPS)
    "vegf2": _complex("vegf2", _VEGF),  # 247
    "il17": _complex("il17", _RAW["IL17_8B7W"]),  # 369
    "tnf2_msa": _complex("tnf2_msa", _RAW["TNF_9BN7"], target_msa=tnf_a3m(), target=_TNF, copies=2),  # 419, real 2225-row MSA on both TNF copies
    "tnf2_nomsa": _complex("tnf2_nomsa", _RAW["TNF_9BN7"], target=_TNF, copies=2),  # 419, same complex, single sequence
    "il17_scfv": _complex("il17_scfv", _RAW["IL17_8DY1"]),  # 499
}

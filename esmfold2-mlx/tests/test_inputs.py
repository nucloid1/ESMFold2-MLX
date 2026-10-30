"""Job files: protein complexes only, rejected loudly otherwise (Review Focus 1)."""

import json

import pytest

from esmfold2_mlx.inputs import Chain, Job, parse_jobs


def _write(tmp_path, jobs):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"jobs": jobs}))
    return str(path)


def _one(tmp_path, **chain):
    return _write(tmp_path, [{"name": "j", "chains": [{"id": "A", "sequence": "MKTAYIAKQR", **chain}]}])


def test_parses_a_complex_with_a_homo_multimer_and_an_msa(tmp_path):
    a3m = tmp_path / "t.a3m"
    a3m.write_text(">q\nMKTAYIAKQR\n")
    jobs = parse_jobs(_write(tmp_path, [{"name": "c1", "chains": [
        {"id": "A", "sequence": "ACDEFGHIKL"}, {"id": ["B", "C"], "sequence": "MKTAYIAKQR", "msa": str(a3m)}]}]))
    assert jobs == [Job("c1", (Chain(("A",), "ACDEFGHIKL", None), Chain(("B", "C"), "MKTAYIAKQR", str(a3m))))]
    assert jobs[0].tokens() == 30


@pytest.mark.parametrize("bad", [{"type": "ligand"}, {"type": "dna"}, {"type": "rna"}, {"smiles": "CCO"}, {"ccd": "ATP"}])
def test_rejects_non_protein_chains(tmp_path, bad):
    with pytest.raises(ValueError, match="protein complexes only"):
        parse_jobs(_one(tmp_path, **bad))


@pytest.mark.parametrize("sequence", ["ACGU", "ACGTUX1", "acdef", "MKT*"])
def test_rejects_sequences_that_are_not_plain_amino_acids(tmp_path, sequence):
    with pytest.raises(ValueError, match="protein complexes only"):
        parse_jobs(_one(tmp_path, sequence=sequence))


def test_rejects_modifications(tmp_path):
    with pytest.raises(ValueError, match="protein complexes only"):
        parse_jobs(_one(tmp_path, modifications=[{"position": 1, "ccd": "SEP"}]))


def test_rejects_duplicate_chain_ids(tmp_path):
    path = _write(tmp_path, [{"name": "j", "chains": [{"id": "A", "sequence": "MKT"}, {"id": ["B", "A"], "sequence": "MKT"}]}])
    with pytest.raises(ValueError, match="duplicate chain id 'A'"):
        parse_jobs(path)


def test_rejects_empty_sequence(tmp_path):
    with pytest.raises(ValueError, match="chain A.*empty"):
        parse_jobs(_one(tmp_path, sequence=""))


def test_msa_path_must_exist(tmp_path):
    with pytest.raises(ValueError, match="no such MSA file"):
        parse_jobs(_one(tmp_path, msa=str(tmp_path / "missing.a3m")))


def test_states_key_depends_on_chains_and_sequences_not_on_msa_or_name(tmp_path):
    a3m = tmp_path / "t.a3m"
    a3m.write_text(">q\nMKT\n")
    base = Job("x", (Chain(("A",), "MKT"),))
    assert base.states_key() == Job("y", (Chain(("A",), "MKT", str(a3m)),)).states_key()
    assert base.states_key() != Job("x", (Chain(("A",), "MKS"),)).states_key()
    assert base.states_key() != Job("x", (Chain(("A", "B"), "MKT"),)).states_key()


def test_rejects_duplicate_job_names_and_missing_chains(tmp_path):
    with pytest.raises(ValueError, match="duplicate job name"):
        parse_jobs(_write(tmp_path, [{"name": "j", "chains": [{"id": "A", "sequence": "MK"}]}] * 2))
    with pytest.raises(ValueError, match="no chains"):
        parse_jobs(_write(tmp_path, [{"name": "j", "chains": []}]))

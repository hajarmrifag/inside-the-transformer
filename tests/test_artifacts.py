import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_executed_results_and_figure_provenance():
    out=ROOT/"results"
    if not (out/"summary.json").exists(): pytest.skip("full run not yet available")
    data=json.loads((out/"dataset.json").read_text());raw=pd.read_csv(out/"interventions.csv")
    base=pd.read_csv(out/"baselines.csv")
    assert len(data)==len(base)==128 and len(raw)==128*12*8
    assert set(raw.id)==set(base.id)=={r["id"] for r in data}
    for name,digest in json.loads((out/"artifact_hashes.json").read_text()).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
    summary=json.loads((out/"summary.json").read_text())
    assert max(summary[k] for k in ["max_noop_error","max_prefix_null_error","max_embedding_repair_error"])<1e-4


def test_frozen_site_selected_only_on_development():
    out=ROOT/"results"
    if not (out/"summary.json").exists(): pytest.skip("full run not yet available")
    data=pd.DataFrame(json.loads((out/"dataset.json").read_text()))
    raw=pd.read_csv(out/"interventions.csv").merge(data,on="id")
    dev=raw[(raw.split=="dev_names")&(raw.position=="final")&(raw.kind=="clean_patch")&(raw.layer>=6)]
    selected=dev.groupby("layer").gain.mean().idxmax()
    assert int(selected)==json.loads((out/"summary.json").read_text())["selected_layer"]


def test_executed_pilot_reproducibility():
    out=ROOT/"results"
    if not (out/"pilot_reproducibility.json").exists(): pytest.skip("two pilot runs not yet available")
    for name,keys,cols in [("baselines",["id"],["clean_ld","corrupted_ld"]),
                          ("interventions",["id","layer","position","kind"],["patched_ld","gain"])]:
        a=pd.read_csv(out/"pilot"/f"{name}.csv").sort_values(keys).reset_index(drop=True)
        b=pd.read_csv(out/"pilot-repeat"/f"{name}.csv").sort_values(keys).reset_index(drop=True)
        assert a[keys].equals(b[keys])
        assert (a[cols]-b[cols]).abs().max().max()<1e-4

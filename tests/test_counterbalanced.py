from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from inside_transformer.counterbalanced import LAYER, baseline_gate
from inside_transformer.counterbalanced_data import build_dataset, candidates, donor_for, validate_balance
from inside_transformer.counterbalanced_analyze import paired_contrasts, stats

ROOT=Path(__file__).resolve().parents[1]


def new_data(model):
    return build_dataset(model,json.loads((ROOT/"results/dataset.json").read_text()))


def test_new_token_alignment_and_complete_answer_boundaries(model):
    rows,ledger=new_data(model)
    assert len(rows)==len(ledger)==128 and all(r["included"] for r in ledger)
    for r in rows:
        c,k=r["clean_tokens"],r["corrupted_tokens"]
        assert len(c)==len(k)
        assert [i for i,(a,b) in enumerate(zip(c,k)) if a!=b]==[r["positions"]["subject"]]
        assert model.tokenizer.decode([r["answer"]])==" "+r["answer_name"]
        assert model.tokenizer.decode([r["foil"]])==" "+r["foil_name"]


def test_factorial_identity_giver_recipient_and_introduction_order():
    rows=candidates();validate_balance(rows)
    for pool in ["evaluation","donor"]:
        subset=[r for r in rows if r["pool"]==pool]
        assert Counter(r["answer_position"] for r in subset)=={1:32,2:32}
        names={r["answer_name"] for r in subset}
        for name in names:
            assert sum(r["giver"]==name for r in subset)==4
            assert sum(r["answer_name"]==name for r in subset)==4
            assert Counter((r["template"],r["answer_position"]) for r in subset if r["answer_name"]==name)=={(0,1):1,(0,2):1,(1,1):1,(1,2):1}


def test_entire_dataset_is_held_out_and_donor_pools_disjoint():
    old=json.loads((ROOT/"results/dataset.json").read_text());new=candidates()
    old_names={r[k] for r in old for k in ["a","b"]}
    old_text={r[k] for r in old for k in ["clean","corrupted"]}
    assert old_names.isdisjoint({r[k] for r in new for k in ["first","second"]})
    assert old_text.isdisjoint({r[k] for r in new for k in ["clean","corrupted"]})
    assert {r["first"] for r in new if r["pool"]=="evaluation"}.isdisjoint({r["first"] for r in new if r["pool"]=="donor"})


def test_exclusive_donor_blocks_and_position_crossing(model):
    rows,_=new_data(model);donors=[r for r in rows if r["pool"]=="donor"]
    for r in [r for r in rows if r["pool"]=="evaluation"]:
        same=donor_for(r,donors,True);opposite=donor_for(r,donors,False)
        assert same["block"]==opposite["block"]==r["block"]
        assert same["answer_position"]==r["answer_position"]
        assert opposite["answer_position"]==3-r["answer_position"]
        assert same["positions"]==opposite["positions"]==r["positions"]


def test_baseline_gate_fails_on_one_bad_position_stratum():
    rows=[dict(pool=p,template=t,answer_position=a,block=0,clean_ld=2.,corrupted_ld=-2.,
               clean_top1_correct=False,corrupted_top1_correct=True)
          for p in ["evaluation","donor"] for t in range(2) for a in [1,2]]
    data=pd.DataFrame(rows)
    assert baseline_gate(data)["all_strata_passed"]  # Pairwise metric; top-1 failure still recorded.
    data.loc[0,"clean_ld"]=-1.
    gate=baseline_gate(data)
    assert not gate["all_strata_passed"] and len(gate["failed_strata"])==1


def test_original_results_remain_byte_identical():
    snapshot=ROOT/"docs/history/position-counterbalanced-v1/previous-results-hashes.json"
    for name,digest in json.loads(snapshot.read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def test_donor_balance_and_paired_contrast_before_cluster_resampling():
    values=[]
    for name in ["a","b"]:
        for kind,gain in [("matched",8.),("unrelated_same_position",6.),("unrelated_opposite_position",0.)]:
            values.append(dict(id=name,kind=kind,gain=gain))
    frame=paired_contrasts(pd.DataFrame(values),pd.DataFrame({"id":["a","b"],"pair":["one","two"]}))
    assert frame.unrelated_balanced.eq(3.).all()
    assert frame.matched_minus_balanced.eq(5.).all()
    result=stats(frame,"matched_minus_balanced")
    assert result["mean"]==result["low"]==result["high"]==5.


def test_new_artifacts_frozen_site_and_controls():
    out=ROOT/"results/counterbalanced-v1"
    if not (out/"summary.json").exists():pytest.skip("extension not yet executed")
    raw=pd.read_csv(out/"interventions.csv");config=json.loads((out/"configuration.json").read_text())
    assert set(raw.layer)=={LAYER}=={10} and config["site_search"] is False
    assert len(raw)==448
    assert set(raw[~raw.kind.isin(["bos_null","name1_null"])].position)=={"final"}
    null=raw[raw.kind.isin(["clean_noop","corrupted_noop","bos_null","name1_null"])]
    assert null.gain.abs().max()<1e-4
    assert json.loads((out/"baseline_validation.json").read_text())["checked_before_interventions"]
    for name,digest in json.loads((out/"artifact_hashes.json").read_text()).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest

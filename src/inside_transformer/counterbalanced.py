"""Frozen-site, position-counterbalanced IOI; refuses to overwrite outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
from transformer_lens import patching

from .core import logit_difference, patched_logits
from .counterbalanced_data import build_dataset, donor_for
from .data import save_json
from .experiment import load_model, seed_everything, MODEL_REVISION

ROOT=Path(__file__).resolve().parents[2]
SEED=20261114
LAYER=10
HOOK=f"blocks.{LAYER}.hook_resid_pre"
KINDS=["matched","unrelated_same_position","unrelated_opposite_position",
       "corrupted_noop","clean_noop","bos_null","name1_null"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_gate(frame):
    strata=[]
    for (pool,template,pos),g in frame.groupby(["pool","template","answer_position"]):
        clean=float((g.clean_ld>0).mean());corrupt=float((g.corrupted_ld<0).mean())
        gap=float((g.clean_ld-g.corrupted_ld).mean())
        strata.append(dict(pool=pool,template=int(template),answer_position=int(pos),n=len(g),
            clusters=int(g.block.nunique()),clean_pairwise_accuracy=clean,corrupted_pairwise_accuracy=corrupt,
            clean_top1_accuracy=float(g.clean_top1_correct.mean()),corrupted_top1_accuracy=float(g.corrupted_top1_correct.mean()),
            mean_clean_ld=float(g.clean_ld.mean()),mean_corrupted_ld=float(g.corrupted_ld.mean()),mean_gap=gap,
            passed=clean>=.8 and corrupt>=.8 and gap>.5))
    return dict(checked_before_interventions=True,all_strata_passed=all(r["passed"] for r in strata),
        strata=strata,failed_strata=[r for r in strata if not r["passed"]])


def measure(logits, rows, model):
    device=logits.device
    a=torch.tensor([r["answer"] for r in rows],device=device)
    b=torch.tensor([r["foil"] for r in rows],device=device)
    last=logits[:,-1];prob=last.softmax(-1);idx=torch.arange(len(rows),device=device)
    ld=logit_difference(logits,a,b).cpu().tolist();top=last.argmax(-1).cpu().tolist()
    pa=prob[idx,a].cpu().tolist();pb=prob[idx,b].cpu().tolist()
    return [dict(ld=ld[i],top1_token=top[i],top1_text=model.tokenizer.decode([top[i]]),
                 clean_answer_probability=pa[i],foil_probability=pb[i]) for i in range(len(rows))]


def device_check(model, rows):
    sample=rows[:2]
    def forward(device):
        model.to(device)
        c=torch.tensor([r["clean_tokens"] for r in sample],device=device)
        k=torch.tensor([r["corrupted_tokens"] for r in sample],device=device)
        cl,cache=model.run_with_cache(c,names_filter=lambda name:name==HOOK)
        outputs=[cl,model(k),patched_logits(model,k,cache,LAYER,-1)]
        a=torch.tensor([r["answer"] for r in sample],device=device)
        b=torch.tensor([r["foil"] for r in sample],device=device)
        return torch.stack([logit_difference(x,a,b).cpu() for x in outputs])
    cpu=forward("cpu")
    record=dict(site=HOOK,position="final",cpu_ld=cpu.tolist(),mps_available=torch.backends.mps.is_available(),selected="cpu")
    if record["mps_available"]:
        try:
            mps=forward("mps");error=float((mps-cpu).abs().max())
            record.update(mps_ld=mps.tolist(),max_abs_ld_error=error)
            if error<=.02: record["selected"]="mps"
            else: record["fallback_reason"]="LD disagreement exceeds 0.02"
        except Exception as exc:
            record["fallback_reason"]=repr(exc)
    model.to(record["selected"])
    return record


def collect_baselines(model, rows):
    records=[];clean_cache={};corrupt_cache={}
    device=model.cfg.device
    for template in range(2):
        subset=[r for r in rows if r["template"]==template]
        for offset in range(0,len(subset),16):
            batch=subset[offset:offset+16]
            assert len({len(r["clean_tokens"]) for r in batch})==1
            record=[dict(id=r["id"]) for r in batch]
            for key,kind,store in [("clean_tokens","clean",clean_cache),("corrupted_tokens","corrupted",corrupt_cache)]:
                tok=torch.tensor([r[key] for r in batch],device=device)
                logits,cache=model.run_with_cache(tok,names_filter=lambda name:name==HOOK)
                values=measure(logits,batch,model)
                for i,(r,v) in enumerate(zip(batch,values)):
                    store[r["id"]]=cache[HOOK][i:i+1].clone()
                    record[i].update({f"{kind}_{k}":val for k,val in v.items()})
                    record[i][f"{kind}_top1_correct"]=v["top1_token"]==r["answer" if kind=="clean" else "foil"]
            records.extend(record)
    return pd.DataFrame(records),clean_cache,corrupt_cache


def official_check(model, row, clean_cache):
    tok=torch.tensor([row["corrupted_tokens"]],device=model.cfg.device)
    a=torch.tensor([row["answer"]],device=model.cfg.device);b=torch.tensor([row["foil"]],device=model.cfg.device)
    cache={HOOK:clean_cache[row["id"]]}
    metric=lambda logits:logit_difference(logits,a,b).mean()
    index=pd.DataFrame([[LAYER,row["positions"]["final"]]],columns=["layer","pos"])
    official=patching.generic_activation_patch(model,tok,cache,metric,patching.layer_pos_patch_setter,"resid_pre",index_df=index)
    custom=metric(patched_logits(model,tok,cache,LAYER,-1))
    error=abs(official.item()-custom.item());assert error<1e-4
    return dict(layer=LAYER,position="final",official_custom_absolute_error=error)


def run_interventions(model, rows, base, clean_cache, corrupt_cache):
    donors=[r for r in rows if r["pool"]=="donor"]
    evaluation=[r for r in rows if r["pool"]=="evaluation"]
    baseline=base.set_index("id")
    records=[]
    for template in range(2):
        subset=[r for r in evaluation if r["template"]==template]
        for offset in range(0,len(subset),16):
            batch=subset[offset:offset+16]
            for kind in KINDS:
                donor_rows=[donor_for(r,donors,same_position=kind=="unrelated_same_position") for r in batch] if kind.startswith("unrelated") else batch
                cache_source=corrupt_cache if kind=="corrupted_noop" else clean_cache
                cache={HOOK:torch.cat([cache_source[r["id"]] for r in donor_rows],dim=0)}
                tokenkey="clean_tokens" if kind=="clean_noop" else "corrupted_tokens"
                tok=torch.tensor([r[tokenkey] for r in batch],device=model.cfg.device)
                position="bos" if kind=="bos_null" else "name1" if kind=="name1_null" else "final"
                pos=batch[0]["positions"][position]
                assert all(r["positions"][position]==pos for r in batch)
                logits=patched_logits(model,tok,cache,LAYER,pos)
                values=measure(logits,batch,model)
                for row,donor,value in zip(batch,donor_rows,values):
                    reference=baseline.loc[row["id"],"clean_ld" if kind=="clean_noop" else "corrupted_ld"]
                    records.append(dict(id=row["id"],donor_id=donor["id"],kind=kind,layer=LAYER,position=position,
                        patched_ld=value.pop("ld"),**value,reference_ld=float(reference)))
                    records[-1]["gain"]=records[-1]["patched_ld"]-float(reference)
                    records[-1]["clean_answer_top1"]=records[-1]["top1_token"]==row["answer"]
    return pd.DataFrame(records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT/"results/counterbalanced-v1")
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    seed_everything();torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter()
    prior_path=ROOT/"results/summary.json";prior=json.loads(prior_path.read_text())
    assert prior["selected_layer"]==LAYER, "prior development-selected site changed"
    save_json(out/"frozen_prior_summary.json",prior)
    config=dict(protocol="docs/position-counterbalanced-protocol.md",protocol_sha256=sha(ROOT/"docs/position-counterbalanced-protocol.md"),
        seed=SEED,layer=LAYER,position="final",hook=HOOK,model="gpt2",model_revision=MODEL_REVISION,
        prior_summary_sha256=sha(prior_path),site_search=False,planned_evaluation_n=64,planned_donor_n=64,
        bootstrap_repeats=5000,cluster="exclusive evaluation+donor name-pair block",intervention_kinds=KINDS)
    save_json(out/"configuration.json",config)
    model=load_model(local_files_only=True)
    rows,ledger=build_dataset(model,json.loads((ROOT/"results/dataset.json").read_text()))
    save_json(out/"candidate_audit.json",ledger);save_json(out/"dataset.json",rows)
    save_json(out/"exclusions.json",[r for r in ledger if not r["included"]])
    assert len({r["block"] for r in rows})>=4
    for row in rows:
        if row["pool"]=="evaluation":
            for same in [True,False]:donor_for(row,[r for r in rows if r["pool"]=="donor"],same)
    device=device_check(model,rows);save_json(out/"device_validation.json",device)
    print("Device:",json.dumps(device),flush=True)
    base,clean_cache,corrupt_cache=collect_baselines(model,rows)
    base.to_csv(out/"baselines.csv",index=False)
    combined=base.merge(pd.DataFrame(rows),on="id",validate="one_to_one")
    gate=baseline_gate(combined);save_json(out/"baseline_validation.json",gate)
    failures=combined[(combined.clean_ld<=0)|(combined.corrupted_ld>=0)|(~combined.clean_top1_correct)|(~combined.corrupted_top1_correct)]
    failures.to_csv(out/"baseline_failures.csv",index=False)
    print("Baseline gate BEFORE interventions:",gate["all_strata_passed"],"failure rows:",len(failures),flush=True)
    save_json(out/"helper_validation.json",official_check(model,next(r for r in rows if r["pool"]=="evaluation"),clean_cache))
    effects=run_interventions(model,rows,base,clean_cache,corrupt_cache)
    effects.to_csv(out/"interventions.csv",index=False)
    null=effects[effects.kind.isin(KINDS[3:])].gain.abs().max()
    assert null<1e-4,f"null control failed: {null}"
    assert sha(prior_path)==config["prior_summary_sha256"]
    save_json(out/"metadata.json",dict(started_utc=started,elapsed_seconds=time.perf_counter()-timer,
        device=device["selected"],dtype="float32",python=platform.python_version(),platform=platform.platform(),
        evaluation_n=sum(r["pool"]=="evaluation" for r in rows),donor_n=sum(r["pool"]=="donor" for r in rows),
        clusters=len({r["block"] for r in rows}),n_interventions=len(effects),max_null_error=float(null),
        baseline_gate_passed=gate["all_strata_passed"],
        versions={k:importlib.metadata.version(k) for k in ["torch","transformer-lens","transformers","numpy","pandas","matplotlib"]},
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"src/inside_transformer").glob("*.py")}))
    print(f"Completed {len(effects)} interventions; outputs: {out}",flush=True)


if __name__=="__main__":main()

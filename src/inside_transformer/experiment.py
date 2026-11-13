"""Run with python -m inside_transformer.experiment --output results."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import platform
import random
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, patching
from transformers import AutoTokenizer

from .core import logit_difference, patched_logits, residual_filter
from .data import POSITIONS, build_dataset, save_json

SEED = 20261113
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def load_model(local_files_only=False):
    tokenizer=AutoTokenizer.from_pretrained("gpt2", revision=MODEL_REVISION,
        add_bos_token=True, local_files_only=local_files_only)
    model=HookedTransformer.from_pretrained("gpt2",device="cpu",dtype=torch.float32,
        revision=MODEL_REVISION,tokenizer=tokenizer,local_files_only=local_files_only)
    model.eval()
    return model


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)


def tokens(rows, key, device):
    return torch.tensor([r[key] for r in rows], device=device)


def evaluate_device(model, rows):
    """Validate MPS baseline AND interventions, benchmark representative forwards."""
    rows = rows[:2]
    record = dict(mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available())
    c, k = [tokens(rows, key, "cpu") for key in ["clean_tokens", "corrupted_tokens"]]
    a = torch.tensor([r["answer"] for r in rows]); b = torch.tensor([r["foil"] for r in rows])
    cl, cache = model.run_with_cache(c, names_filter=residual_filter)
    kl = model(k)
    cp = patched_logits(model, k, cache, 6, -1)
    cpu = torch.stack([logit_difference(x,a,b) for x in [cl,kl,cp]])
    start=time.perf_counter()
    for _ in range(3): model(k)
    record["cpu_seconds_per_forward"]=(time.perf_counter()-start)/3
    record["cpu_logit_differences"]=cpu.tolist()
    record["selected"]="cpu"
    if record["mps_available"]:
        try:
            model.to("mps"); c=c.to("mps"); k=k.to("mps"); a=a.to("mps"); b=b.to("mps")
            ml, mc = model.run_with_cache(c,names_filter=residual_filter)
            mk=model(k); mp=patched_logits(model,k,mc,6,-1)
            mps=torch.stack([logit_difference(x,a,b) for x in [ml,mk,mp]]).cpu()
            error=(mps-cpu).abs().max().item()
            record["mps_logit_differences"]=mps.tolist(); record["max_ld_error"]=error
            torch.mps.synchronize(); start=time.perf_counter()
            for _ in range(3): model(k)
            torch.mps.synchronize()
            record["mps_seconds_per_forward"]=(time.perf_counter()-start)/3
            if error <= .02:
                record["selected"]="mps"
            else:
                record["fallback_reason"]="CPU/MPS LD disagreement exceeds 0.02"
        except Exception as exc:
            record["fallback_reason"]=repr(exc)
    model.to(record["selected"])
    return record


def validate_official_helper(model, row):
    device=model.cfg.device
    c=tokens([row],"clean_tokens",device); k=tokens([row],"corrupted_tokens",device)
    _,cache=model.run_with_cache(c,names_filter=residual_filter)
    a=torch.tensor([row["answer"]],device=device); b=torch.tensor([row["foil"]],device=device)
    index=pd.DataFrame([[6,len(row["clean_tokens"])-1]],columns=["layer","pos"])
    metric=lambda logits: logit_difference(logits,a,b).mean()
    official=patching.generic_activation_patch(model,k,cache,metric,
        patching.layer_pos_patch_setter,"resid_pre",index_df=index)
    custom=metric(patched_logits(model,k,cache,6,-1))
    error=(official.reshape(-1)[0]-custom).abs().item()
    assert error < 1e-4, f"official helper mismatch {error}"
    return {"official_custom_absolute_error":error,"official_ld":official.item(),
            "custom_ld":custom.item(),"api_signature":str(inspect.signature(patching.generic_activation_patch))}


def run_batches(model, rows, out):
    baselines, interventions=[],[]
    device=model.cfg.device
    groups={}
    for row in rows: groups.setdefault((row["pool"],row["template"]),[]).append(row)
    for gi,(group,batch) in enumerate(groups.items()):
        print(f"Batch {gi+1}/{len(groups)} {group}, n={len(batch)}",flush=True)
        assert len({len(r["clean_tokens"]) for r in batch})==1
        c=tokens(batch,"clean_tokens",device); k=tokens(batch,"corrupted_tokens",device)
        a=torch.tensor([r["answer"] for r in batch],device=device)
        b=torch.tensor([r["foil"] for r in batch],device=device)
        cl,cc=model.run_with_cache(c,names_filter=residual_filter)
        kl,kc=model.run_with_cache(k,names_filter=residual_filter)
        cd=logit_difference(cl,a,b).cpu().numpy(); kd=logit_difference(kl,a,b).cpu().numpy()
        cp=cl[:,-1].argmax(-1).cpu().tolist(); kp=kl[:,-1].argmax(-1).cpu().tolist()
        probc=cl[:,-1].softmax(-1); probk=kl[:,-1].softmax(-1)
        for i,r in enumerate(batch):
            baselines.append(dict(id=r["id"],clean_ld=float(cd[i]),corrupted_ld=float(kd[i]),
                clean_top1_token=cp[i],corrupted_top1_token=kp[i],
                clean_top1_text=model.tokenizer.decode([cp[i]]),corrupted_top1_text=model.tokenizer.decode([kp[i]]),
                clean_top1_correct=cp[i]==r["answer"],corrupted_top1_correct=kp[i]==r["foil"],
                clean_answer_probability=probc[i,a[i]].item(),corrupted_foil_probability=probk[i,b[i]].item()))
        del cl,kl,probc,probk
        # Each pair has adjacent opposite orders; +2 picks a disjoint name pair.
        donor_indices=[(i+2)%len(batch) for i in range(len(batch))]
        for i,j in enumerate(donor_indices):
            assert {batch[i]["a"],batch[i]["b"]}.isdisjoint({batch[j]["a"],batch[j]["b"]})
        wrong={name:val[donor_indices] for name,val in cc.items()}
        for layer in range(model.cfg.n_layers):
            for position in POSITIONS:
                pos=batch[0]["positions"][position]
                assert all(r["positions"][position]==pos for r in batch)
                ld=logit_difference(patched_logits(model,k,cc,layer,pos),a,b).cpu().numpy()
                for i,r in enumerate(batch):
                    interventions.append(dict(id=r["id"],layer=layer,position=position,kind="clean_patch",
                                              patched_ld=float(ld[i]),gain=float(ld[i]-kd[i]),donor_id=r["id"]))
            for kind,recipient,cache,base in [("corrupted_noop",k,kc,kd),("clean_noop",c,cc,cd),("unrelated_donor",k,wrong,kd)]:
                ld=logit_difference(patched_logits(model,recipient,cache,layer,-1),a,b).cpu().numpy()
                for i,r in enumerate(batch):
                    interventions.append(dict(id=r["id"],layer=layer,position="final",kind=kind,
                        patched_ld=float(ld[i]),gain=float(ld[i]-base[i]),
                        donor_id=batch[donor_indices[i]]["id"] if kind=="unrelated_donor" else r["id"]))
        pd.DataFrame(baselines).to_csv(out/"baselines.csv",index=False)
        pd.DataFrame(interventions).to_csv(out/"interventions.csv",index=False)
        del cc,kc,wrong
    return baselines,interventions


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=Path("results"))
    p.add_argument("--pilot",action="store_true"); p.add_argument("--device",choices=["auto","cpu","mps"],default="auto")
    args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    seed_everything(); start=time.perf_counter(); started_utc=datetime.now(timezone.utc).isoformat()
    save_json(out/"configuration.json",dict(seed=SEED,model="gpt2",model_revision=MODEL_REVISION,
        dtype="float32",requested_device=args.device,pilot=args.pilot,positions=POSITIONS,
        layers=list(range(12)),max_batch_size=16,selection_layers=list(range(6,12)),
        selection_metric="mean raw final-position LD gain on dev_names only",bootstrap_clusters="unordered name pair",
        bootstrap_repeats=5000,exclusion_policy="tokenization and alignment only"))
    revision=Path(hf_hub_download("gpt2","config.json", revision=MODEL_REVISION)).parent.name
    print(f"Loading gpt2 revision {revision}",flush=True)
    model=load_model()
    rows,excluded=build_dataset(model)
    save_json(out/"dataset.json",rows); save_json(out/"exclusions.json",excluded)
    assert len(rows)>0
    dev=[r for r in rows if r["split"]=="dev_names" and r["template"]==0]
    check=evaluate_device(model,dev)
    if args.device!="auto":
        if args.device=="mps": assert check["selected"]=="mps", "MPS not validated"
        model.to(args.device); check["selected"]=args.device
    save_json(out/"device_validation.json",check)
    print(json.dumps(check,indent=2),flush=True)
    helper=validate_official_helper(model,dev[0]); save_json(out/"helper_validation.json",helper)
    if args.pilot: rows=dev[:4]; save_json(out/"dataset.json",rows)
    run_batches(model,rows,out)
    meta=dict(started_utc=started_utc,elapsed_seconds=time.perf_counter()-start,
        seed=SEED,pilot=args.pilot,device=check["selected"],model="gpt2",model_revision=revision,
        dtype="float32",n_examples=len(rows),n_excluded=len(excluded),platform=platform.platform(),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python=platform.python_version(),machine=platform.machine(),n_layers=model.cfg.n_layers,
        versions={k:importlib.metadata.version(k) for k in ["torch","transformer-lens","transformers","numpy","pandas","matplotlib"]},
        model_processing=dict(fold_ln=True,center_writing_weights=True,center_unembed=True),
        source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path("src").rglob("*.py")})
    save_json(out/"metadata.json",meta)
    print(f"Finished {len(rows)} examples in {meta['elapsed_seconds']:.1f}s",flush=True)


if __name__=="__main__": main()

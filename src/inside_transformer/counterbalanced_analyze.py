"""Paired, block-clustered summaries of the frozen-site extension."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analyze import cluster_ci
from .counterbalanced import ROOT, SEED, KINDS, LAYER, sha
from .data import save_json


def paired_contrasts(raw, data):
    wide=raw.pivot(index="id",columns="kind",values="gain").reset_index()
    wide["unrelated_balanced"]=(wide.unrelated_same_position+wide.unrelated_opposite_position)/2
    wide["matched_minus_balanced"]=wide.matched-wide.unrelated_balanced
    wide["matched_minus_same"]=wide.matched-wide.unrelated_same_position
    wide["matched_minus_opposite"]=wide.matched-wide.unrelated_opposite_position
    wide["same_minus_opposite"]=wide.unrelated_same_position-wide.unrelated_opposite_position
    return wide.merge(data,on="id",validate="one_to_one")


def stats(frame,column):
    return cluster_ci(frame,column,seed=SEED,repeats=5000)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT/"results/counterbalanced-v1")
    args=parser.parse_args();out=args.output.resolve()
    data=pd.DataFrame(json.loads((out/"dataset.json").read_text()))
    base=pd.read_csv(out/"baselines.csv").merge(data,on="id",validate="one_to_one")
    raw=pd.read_csv(out/"interventions.csv").merge(data,on="id",validate="many_to_one")
    evalbase=base[base.pool=="evaluation"]
    assert len(raw)==len(evalbase)*7
    assert raw.groupby("id").kind.nunique().eq(7).all() and set(raw.kind)==set(KINDS)
    assert raw.layer.eq(LAYER).all()
    gate=json.loads((out/"baseline_validation.json").read_text())
    paired=paired_contrasts(raw,evalbase[["id","pair","block","template","answer_position"]])
    paired.to_csv(out/"paired_contrasts.csv",index=False)
    gap=float((evalbase.clean_ld-evalbase.corrupted_ld).mean())
    aggregate=[]
    groups=[("overall","all",raw)]
    groups += [("answer_position",str(pos),g) for pos,g in raw.groupby("answer_position")]
    groups += [("template_position",f"t{t}-p{pos}",g) for (t,pos),g in raw.groupby(["template","answer_position"])]
    for group,label,frame in groups:
        for kind,g in frame.groupby("kind"):
            aggregate.append(dict(group=group,label=label,kind=kind,**stats(g,"gain"),
                clean_pairwise_preference=float((g.patched_ld>0).mean()),
                clean_answer_top1_accuracy=float(g.clean_answer_top1.mean())))
    agg=pd.DataFrame(aggregate);agg.to_csv(out/"aggregated.csv",index=False)
    contrasts={name:stats(paired,name) for name in ["matched_minus_balanced","matched_minus_same","matched_minus_opposite","same_minus_opposite","unrelated_balanced"]}
    by_position=[]
    for pos in [1,2]:
        frame=raw[(raw.answer_position==pos)&(raw.kind=="matched")]
        by_position.append(dict(answer_position=pos,**stats(frame,"gain")))
    baseline_summary=[]
    for column in ["clean_ld","corrupted_ld"]:
        baseline_summary.append(dict(metric=column,**stats(evalbase,column)))
    summary=dict(layer=LAYER,position="final",evaluation_n=len(evalbase),donor_n=int((data.pool=="donor").sum()),
        clusters=int(evalbase.block.nunique()),baseline_gate_passed=gate["all_strata_passed"],
        baseline_means=baseline_summary,mean_baseline_gap=gap,
        baseline_clean_pairwise_accuracy=float((evalbase.clean_ld>0).mean()),
        baseline_corrupted_pairwise_accuracy=float((evalbase.corrupted_ld<0).mean()),
        baseline_clean_top1_accuracy=float(evalbase.clean_top1_correct.mean()),
        baseline_corrupted_top1_accuracy=float(evalbase.corrupted_top1_correct.mean()),
        conditions=[dict(kind=kind,**stats(g,"gain"),clean_pairwise_preference=float((g.patched_ld>0).mean()),
            clean_answer_top1_accuracy=float(g.clean_answer_top1.mean()),
            recovery=float(g.gain.mean()/gap) if gate["all_strata_passed"] and kind in KINDS[:3] else None)
            for kind,g in raw.groupby("kind")],
        matched_by_answer_position=by_position,contrasts=contrasts,
        hypothesis_supported=bool(gate["all_strata_passed"] and all(r["low"]>0 for r in by_position) and contrasts["matched_minus_balanced"]["low"]>0),
        max_null_error=float(raw[raw.kind.isin(KINDS[3:])].gain.abs().max()),
        bootstrap_repeats=5000,seed=SEED)
    save_json(out/"summary.json",summary)
    figures=out/"figures";figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    x=np.arange(4);labels=["T0 · first","T0 · second","T1 · first","T1 · second"]
    for offset,column,label,color in [(-.14,"clean_ld","Clean","#30638e"),(.14,"corrupted_ld","Counterfactual","#d16b45")]:
        values=[stats(evalbase[(evalbase.template==t)&(evalbase.answer_position==pos)],column) for t in [0,1] for pos in [1,2]]
        m=np.array([r["mean"] for r in values]);e=np.array([m-[r["low"] for r in values],[r["high"] for r in values]-m])
        axes[0].errorbar(x+offset,m,yerr=e,fmt="o",capsize=3,label=label,color=color)
    axes[0].set_xticks(x,labels,rotation=25);axes[0].axhline(0,color="black",lw=.6)
    axes[0].set(ylabel="Clean-answer minus counterfactual-answer logit",xlabel="Template and clean-answer introduction position",title="Unmodified evaluation predictions\nn=16 per point; 8 pair blocks; 95% bootstrap")
    axes[0].legend()
    evalgate=[g for g in gate["strata"] if g["pool"]=="evaluation"]
    for offset,key,label,color in [(-.15,"clean_pairwise_accuracy","Clean pairwise","#30638e"),(.15,"corrupted_pairwise_accuracy","Counterfactual pairwise","#d16b45")]:
        axes[1].bar(x+offset,[g[key] for g in evalgate],width=.29,label=label,color=color)
    axes[1].set_xticks(x,labels,rotation=25);axes[1].set(ylim=(0,1.12),ylabel="Pairwise accuracy",title="Baseline gate (before experimental patches)\nThreshold 0.80 in every evaluation and donor cell")
    axes[1].axhline(.8,ls="--",lw=.8,color="black");axes[1].legend(fontsize=8)
    fig.savefig(figures/"baseline.png",dpi=170,bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    mainkinds=KINDS[:3];colors=["#30638e","#d16b45","#459477"]
    for kind,offset,label,color in zip(mainkinds,[-.18,0,.18],["Matched","Unrelated: same answer position","Unrelated: opposite answer position"],colors):
        values=[stats(raw[(raw.kind==kind)&(raw.answer_position==pos)],"gain") for pos in [1,2]]
        m=np.array([r["mean"] for r in values]);e=np.array([m-[r["low"] for r in values],[r["high"] for r in values]-m])
        axes[0].errorbar(np.arange(2)+offset,m,yerr=e,fmt="o",capsize=4,label=label,color=color)
    axes[0].set_xticks([0,1],["First introduced","Second introduced"])
    axes[0].set(ylabel="Raw logit-difference gain",xlabel="Clean-answer introduction position",title="Fixed block 10 input · final token\nn=32 per point; 8 pair blocks; 95% bootstrap")
    axes[0].axhline(0,color="black",lw=.6);axes[0].legend(fontsize=8)
    names=["Matched − balanced donor","Same − opposite donor","Counterfactual self","Clean self","BOS null","Name-1 null"]
    values=[contrasts["matched_minus_balanced"],contrasts["same_minus_opposite"]]
    values += [stats(raw[raw.kind==kind],"gain") for kind in KINDS[3:]]
    m=np.array([r["mean"] for r in values]);e=np.array([m-[r["low"] for r in values],[r["high"] for r in values]-m])
    axes[1].errorbar(range(6),m,yerr=e,fmt="o",capsize=4,color="#30638e")
    axes[1].set_xticks(range(6),names,rotation=35,ha="right");axes[1].axhline(0,color="black",lw=.6)
    axes[1].set(ylabel="Paired contrast or null gain (logit units)",title="Donor-position diagnostic and controls\nn=64 per contrast; 8 exclusive donor+recipient blocks")
    fig.savefig(figures/"effects_and_controls.png",dpi=170,bbox_inches="tight");plt.close(fig)
    paths=[p for p in out.glob("*.json") if p.name!="artifact_hashes.json"]+list(out.glob("*.csv"))+list(figures.glob("*.png"))
    save_json(out/"artifact_hashes.json",{str(p.relative_to(out)):sha(p) for p in sorted(paths)})
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()

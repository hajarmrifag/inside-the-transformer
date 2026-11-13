"""Deterministic aggregation and plots sourced exclusively from raw CSV files."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import POSITIONS, save_json
from .experiment import SEED

SPLITS=["dev_names","dev_new_names","new_template_names","new_template_new_names"]
LABELS=["Development","New names","New templates","Both held out"]


def cluster_ci(frame, column="gain", seed=SEED, repeats=5000):
    means=frame.groupby("pair")[column].mean().to_numpy()
    rng=np.random.default_rng(seed)
    draws=rng.choice(means,size=(repeats,len(means)),replace=True).mean(1)
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(mean=float(frame[column].mean()),low=float(lo),high=float(hi),
                n=len(frame),clusters=len(means))


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("results"));args=p.parse_args()
    out=args.output;figdir=out/"figures";figdir.mkdir(exist_ok=True)
    data=pd.DataFrame(json.loads((out/"dataset.json").read_text()))
    base=pd.read_csv(out/"baselines.csv").merge(data,on="id",validate="one_to_one")
    raw=pd.read_csv(out/"interventions.csv").merge(data,on="id",validate="many_to_one")
    assert len(base)==len(data) and len(raw)==len(data)*12*8
    final=raw[(raw.kind=="clean_patch")&(raw.position=="final")]
    discovery=final[(final.split=="dev_names")&(final.layer>=6)]
    selected=int(discovery.groupby("layer").gain.mean().idxmax())
    chosen=raw[(raw.layer==selected)&(raw.position=="final")]
    summaries=[]
    for split in SPLITS:
        b=base[base.split==split]; f=chosen[(chosen.split==split)&(chosen.kind=="clean_patch")]
        wrong=chosen[(chosen.split==split)&(chosen.kind=="unrelated_donor")][["id","gain"]]
        paired=f.merge(wrong,on="id",suffixes=("","_wrong"));paired["specificity"]=paired.gain-paired.gain_wrong
        gap=float((b.clean_ld-b.corrupted_ld).mean())
        summaries.append(dict(split=split,**cluster_ci(f),specificity=cluster_ci(paired,"specificity"),
            clean_ld=cluster_ci(b,"clean_ld"),corrupted_ld=cluster_ci(b,"corrupted_ld"),gap=gap,
            recovery=float(f.gain.mean()/gap) if gap>.5 else None,
            clean_pairwise_accuracy=float((b.clean_ld>0).mean()),
            corrupted_pairwise_accuracy=float((b.corrupted_ld<0).mean()),
            clean_top1_accuracy=float(b.clean_top1_correct.mean()),corrupted_top1_accuracy=float(b.corrupted_top1_correct.mean())))
    summary=dict(selected_layer=selected,splits=summaries,bootstrap_repeats=5000,
        hypothesis_supported=bool(summaries[-1]["low"]>0 and summaries[-1]["specificity"]["low"]>0),
        max_noop_error=float(raw[raw.kind.isin(["clean_noop","corrupted_noop"])].gain.abs().max()),
        max_prefix_null_error=float(raw[(raw.kind=="clean_patch")&raw.position.isin(["bos","name1","name2"])].gain.abs().max()))
    positive=raw[(raw.kind=="clean_patch")&(raw.layer==0)&(raw.position=="subject")].merge(base[["id","clean_ld"]],on="id")
    summary["max_embedding_repair_error"]=float((positive.patched_ld-positive.clean_ld).abs().max())
    assert summary["max_noop_error"]<1e-4
    assert summary["max_prefix_null_error"]<1e-4
    assert summary["max_embedding_repair_error"]<1e-4
    save_json(out/"summary.json",summary)
    aggregate=[]
    for (split,layer,pos,kind),f in raw.groupby(["split","layer","position","kind"]):
        aggregate.append(dict(split=split,layer=int(layer),position=pos,kind=kind,**cluster_ci(f)))
    agg=pd.DataFrame(aggregate);agg.to_csv(out/"aggregated.csv",index=False)
    plt.rcParams.update({"font.size":10,"figure.dpi":150,"axes.spines.top":False,"axes.spines.right":False})
    colors=["#30638e","#d16b45","#459477","#845e9d"]
    def finish(fig,name):
        fig.tight_layout();fig.savefig(figdir/f"{name}.png",dpi=170,bbox_inches="tight");plt.close(fig)
    def errors(stats):
        m=np.array([s["mean"] for s in stats]);return m,np.array([m-np.array([s["low"] for s in stats]),np.array([s["high"] for s in stats])-m])
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4)
    for offset,key,label,color in [(-.17,"clean_ld","Clean",colors[0]),(.17,"corrupted_ld","Counterfactual",colors[1])]:
        m,e=errors([s[key] for s in summaries]);ax.bar(x+offset,m,.33,yerr=e,capsize=3,label=label,color=color)
    ax.axhline(0,color="black",lw=.6);ax.set_xticks(x,LABELS);ax.set_ylabel("Logit(clean answer) − logit(counterfactual answer)")
    ax.set_title("Unmodified predictions · n=32 per split, 8 name-pair clusters\n95% cluster-bootstrap intervals");ax.legend();finish(fig,"baseline")
    fig=plt.figure(figsize=(10,5),layout="constrained")
    grid=fig.add_gridspec(1,3,width_ratios=[1,1,.06])
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
    cax=fig.add_subplot(grid[0,2])
    heat=agg[(agg.split.isin([SPLITS[0],SPLITS[-1]]))&(agg.kind=="clean_patch")]
    color_limit=float(np.ceil(heat["mean"].abs().max()))
    for ax,split,title in zip(axes,[SPLITS[0],SPLITS[-1]],["Development","Both held out"]):
        z=agg[(agg.split==split)&(agg.kind=="clean_patch")].pivot(index="layer",columns="position",values="mean").reindex(columns=POSITIONS)
        im=ax.imshow(z,aspect="auto",cmap="RdBu_r",vmin=-color_limit,vmax=color_limit,origin="lower")
        ax.set_xticks(range(5),["BOS","Name 1","Name 2","Subject","Final"],rotation=30);ax.set_yticks(range(12));ax.set_title(title+" · n=32");ax.set_xlabel("Patched token position")
    axes[0].set_ylabel("Block input layer (zero-based)");fig.colorbar(im,cax=cax,label="Mean raw logit-difference gain")
    fig.savefig(figdir/"patching_heatmap.png",dpi=170,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for split,label,color in zip(SPLITS,LABELS,colors):
        f=agg[(agg.split==split)&(agg.kind=="clean_patch")&(agg.position=="final")].sort_values("layer")
        ax.plot(f.layer,f["mean"],marker="o",label=label,color=color);ax.fill_between(f.layer,f.low,f.high,alpha=.12,color=color)
    ax.axvline(selected,color="black",ls="--",lw=.9,label=f"Frozen layer {selected}");ax.axhline(0,color="black",lw=.6)
    ax.set(xlabel="Block input layer (zero-based)",ylabel="Raw logit-difference gain",title="Final-position recovery · n=32 per split\nShading: 95% name-pair cluster bootstrap");ax.legend(fontsize=8,ncol=2);finish(fig,"recovery_layers")
    fig,ax=plt.subplots(figsize=(9,4))
    for pool,offset,color in [("development",-.15,colors[0]),("heldout",.15,colors[1])]:
        stats=[cluster_ci(chosen[(chosen.pool==pool)&(chosen.template==t)&(chosen.kind=="clean_patch")]) for t in range(4)]
        m,e=errors(stats);ax.errorbar(np.arange(4)+offset,m,yerr=e,fmt="o",capsize=4,label=f"{pool} names",color=color)
    ax.set_xticks(range(4),["T0: store","T1: park","T2: office*","T3: garden*"])
    ax.axhline(0,color="black",lw=.6);ax.set(ylabel="Raw logit-difference gain",xlabel="Template (* held out)",title=f"Template robustness at frozen layer {selected} · final token\nn=16 per point, 8 pair clusters; 95% bootstrap intervals");ax.legend();finish(fig,"templates")
    fig,ax=plt.subplots(figsize=(9,4));control_stats=[]
    f=chosen[chosen.split==SPLITS[-1]]
    for kind in ["clean_patch","unrelated_donor","corrupted_noop","clean_noop"]: control_stats.append(cluster_ci(f[f.kind==kind]))
    control_stats.append(cluster_ci(raw[(raw.split==SPLITS[-1])&(raw.layer==selected)&(raw.position=="bos")]))
    m,e=errors(control_stats);ax.bar(range(5),m,yerr=e,capsize=4,color=[colors[0],colors[1],"gray","gray","gray"])
    ax.set_xticks(range(5),["Matched clean","Unrelated clean","Counterfactual self","Clean self","BOS clean"])
    ax.axhline(0,color="black",lw=.6);ax.set(ylabel="Raw logit-difference gain vs recipient",title=f"Both-held-out controls · layer {selected}, n=32\n95% cluster bootstrap; unrelated donor is not guaranteed null");finish(fig,"controls")
    files=[out/"dataset.json",out/"baselines.csv",out/"interventions.csv",out/"summary.json",out/"aggregated.csv",*figdir.glob("*.png")]
    save_json(out/"artifact_hashes.json",{str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()

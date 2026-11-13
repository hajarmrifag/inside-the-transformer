# Inside the Transformer: Causal Tracing of Language Model Decisions

## Abstract

This conceptual replication evaluates whether clean GPT-2-small residual activations restore a paired counterfactual IOI answer preference, with a small frozen-site robustness extension. We executed 128 aligned prompts across four templates and disjoint name pools; no examples were excluded. The development-only rule selected final-token residual input to block 10. On 32 prompts with both unseen names and templates, mean gain was 7.993 logit-difference units (95% name-pair cluster bootstrap [7.334, 8.649]), corresponding to 100.27% of the aggregate clean/counterfactual gap. Matched patches exceeded unrelated donors by 0.950 [0.590, 1.292]. However, unrelated donors recovered 88.3% of that gap, limiting name-specific interpretations. Structural nulls and self-patches were zero-effect, and the embedding repair control exactly reconstructed clean LD. These results support a narrow causal transfer claim, not circuit discovery.

## Background and prior work

Wang et al. established IOI as a natural-language behavior in GPT-2-small and investigated an attention-head circuit using causal methods. This project replicates the general task and residual intervention logic, not their exact dataset, 26/28-head circuit variants, or original faithfulness/completeness/minimality results. ROME supplies a causal-tracing precedent in factual prediction; we perform no ROME weight editing. Zhang and Nanda motivate scrutiny of corruption and metric choices. Our original contribution at project level is the controlled robustness evaluation, not invention of activation patching or a claim of previously unknown circuitry.

AI safety motivation: explanations used for oversight need causal validation and robustness checks. This exercise develops that methodology on a small, auditable behavior. It supplies no direct evidence about advanced-system safety.

## Hypothesis and analysis rule

The protocol in [protocol.md](protocol.md) was written before pilot/model results. Among final-token layers 6–11, select the largest mean raw patch gain using development names AND templates only. Apply that deterministic rule without using held-out outcomes. The complete sweep is collected before aggregation; analytical selection uses only development rows, and held-out sweeps are descriptive. On the both-held-out cell require positive lower bounds for BOTH (1) matched-patch raw gain and (2) paired matched-minus-unrelated-donor gain. Otherwise label the hypothesis unsupported/inconclusive. No post-result layer tuning or example filtering occurred.

Both criteria were met for this constructed sample. This is not a claim of statistical generalization to arbitrary names, templates or models.

## Experimental design

Four templates use store, park, office and garden contexts. T0/T1 are development templates; T2/T3 are held out. Each of two disjoint 16-name pools forms eight disjoint pairs, each used in both orders. The resulting 2 name pools × 4 templates × 8 pairs × 2 orders yields 128 prompts, 32 per names × templates cell. Names and templates are hand chosen. All 128 candidates passed tokenizer/alignment checks; the exclusion log is empty. No model-behavior exclusion was permitted.

Clean prompts introduce A and B, then repeat B as giver; A is the intended answer. Counterfactual prompts change only that repetition to A; B is now intended. Tokenization uses explicit BOS and checks complete answer continuations, single leading-space name tokens, name occurrence counts, one changed token, matched lengths and semantic positions. This counterfactual is grammatical, not random noise.

LD is always logit(A) − logit(B), including counterfactuals. Therefore positive clean LD and negative counterfactual LD both indicate task success. Gain = patched LD − counterfactual LD. Secondary recovery = mean gain / mean clean-minus-counterfactual LD, reported only with mean gap >0.5, without clipping. A value slightly above 100% is an overshoot on this metric, not perfect restoration of every behavior.

We replace one residual vector at `blocks.L.hook_resid_pre`, L=0..11, at BOS, name 1, name 2, repeated subject, and final token. Layer numbering is zero-based; layer 10 means the input to block 10, not its output. At every layer, final-token self-patches on clean/counterfactual runs and an unrelated clean donor add three control interventions per example. Donors rotate by two rows within the same template/pool, ensuring disjoint names. Total interventions: 128 × 12 × (5+3) = 12,288; clean and counterfactual baselines add 256 prompt evaluations. Batch size is 16, float32, no training.

Uncertainty uses 5,000 fixed-seed percentile resamples of unordered name-pair means, preserving both orientations and repeated templates. Each cell has eight clusters. The two primary held-out contrasts are paired; no test treats template copies as independent name samples. Raw measurements and all aggregate site intervals are retained.

## Execution and feasibility

Executed on macOS-26.5-arm64-arm-64bit with Python 3.11.17, PyTorch 2.15.0 and TransformerLens 2.16.1. Current TransformerLens 4.0 has changed model-loading APIs; this project deliberately executes the validated 2.16.1 interface and records the installed patch-helper signature. GPT-2 model and tokenizer revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Standard TransformerLens layer-norm folding/weight centering is enabled and recorded.

MPS was actually available and validated against CPU clean, counterfactual and representative patched LD: maximum absolute difference 0.00001335, below the prespecified 0.02 tolerance. A custom `run_with_hooks` patch matched the official `generic_activation_patch` helper exactly at the validation site. Two four-example pilots yielded identical baseline and all 384 intervention measurements. Two full runs also matched within the recorded tolerance (see `results/full_reproducibility.json`). The final experiment took 30.7 seconds after cached model loading; the initial model download was additional. Peak process RSS was 1.57 GiB; this is not a complete unified-memory accounting. Model weights are cached outside the repository.

## Results

| Evaluation split | Prompts / pairs | Clean LD | Counterfactual LD | Patch gain [95% CI] | Gap recovery |
|---|---:|---:|---:|---:|---:|
| Development | 32 / 8 | 4.141 | -4.325 | 8.627 [7.423, 9.762] | 101.90% |
| New names | 32 / 8 | 4.305 | -5.012 | 9.636 [8.947, 10.316] | 103.42% |
| New templates | 32 / 8 | 4.031 | -3.256 | 7.091 [6.170, 7.966] | 97.30% |
| Both held out | 32 / 8 | 4.371 | -3.601 | 7.993 [7.334, 8.649] | 100.27% |

All cells contain 32 prompts but eight pair clusters. Clean and grammatical counterfactual unmodified predictions were correct on all 128 prompts each, for both two-name preference and full-vocabulary top-1. These are easy constructed prompts; this ceiling is not evidence of general language competence. Every candidate was retained.

On the both-held-out cell, matched patches restored the clean answer's pairwise preference in 100% of examples. The matched-minus-unrelated paired contrast was 0.950 [0.590, 1.292]. Yet unrelated donors had mean gain 7.043, or 88.3% recovery. Most of the aggregate restoration therefore does not require a donor with the same name identities. This does not isolate whether the transferable content is positional, role-based, task-general, or a mixture.

All self-patch gains and all pre-change-position patch gains were exactly zero in the executed measurements. The layer-0 subject patch matched clean LD exactly. These validate intervention implementation and the predicted causal nulls. They do not by themselves validate a complete mechanistic explanation.

![Clean versus counterfactual predictions](../results/figures/baseline.png)

![Residual patching heatmap](../results/figures/patching_heatmap.png)

![Recovery across layers](../results/figures/recovery_layers.png)

![Template transfer](../results/figures/templates.png)

![Intervention controls](../results/figures/controls.png)

Figures are computed directly from saved raw CSV files. Error bars/shading are pointwise 95% pair-cluster bootstrap intervals. The heatmap shows means only, with an unclipped shared color range; corresponding uncertainty is in `results/aggregated.csv`. All-layer curves are descriptive after site selection. Hashes tie figures and aggregates to raw files.

## Interpretation

Late final-token residual states can causally shift the counterfactual prediction toward its clean paired answer. A layer selected by the development-only rule retains this effect across the held-out samples. Positive matched-versus-unrelated contrast gives limited donor-specific evidence, while the large unrelated-donor effect and fixed first-name clean answer substantially weaken identity-specific claims. The layer-zero positive check is input repair. None of these results establishes a complete IOI circuit, necessity of a component, uniqueness of a representation, or safety of a language model.

## Limitations

- Only four hand-written templates and eight name-pair clusters per evaluation cell. The held-out templates retain a closely related grammatical structure; this is a small transfer check, not broad linguistic generalization.
- Both name orders are counterbalanced, but the clean answer is always the first introduced name and the counterfactual answer is always the second. Answer-position and clean/counterfactual role are confounded. Transfer could use generic positional/role information; the strong unrelated-donor effect is consistent with this possibility, but does not establish its mechanism.
- Entire 768-dimensional residual vectors are transplanted. Late final-position states already include much of the computation leading to the answer. Recovery cannot identify a complete circuit, a unique semantic feature, necessity, or exclusivity. No head-level or path-patching analysis was performed.
- A wrong-name clean donor is not a guaranteed causal null. Its substantial recovery makes a simple “only the correct-name state restores behavior” story untenable.
- Unrelated donors use one fixed cyclic assignment. Bootstrap resampling conditions on this mapping and does not integrate donor-selection uncertainty; dependencies involving shared donor/recipient pairs further limit population-level inference.
- Layer-0 subject repair is an input reconstruction check. It must not be presented as a nontrivial discovered mechanism.
- Bootstrap intervals use only eight fixed, manually chosen pair clusters per cell, with both orders and repeated templates kept together. They are conditional sensitivity estimates; nominal 95% coverage for arbitrary English is not established. Layer-sweep intervals are descriptive and pointwise, not simultaneous.
- Development effects are selected and optimistic. Only the prespecified held-out contrasts test transfer. Normalized recovery is a ratio of means without a reported interval; raw gains carry the primary uncertainty estimates.
- Logit differences compare two names. Unmodified full-vocabulary top-1 is also reported, but patched full-vocabulary top-1 was not retained; restored pairwise preference must not be described as measured patched top-1 accuracy.
- No pretrained-data contamination audit, demographic balancing, alternative corruption distribution, second model, head circuit replication, or cross-machine replication was performed. GPT-2 IOI does not establish deployment safety, honesty, or deception detection.


## Future research

Preregister a larger, independently generated evaluation with both clean answer positions, more pairs and more structurally different templates. Test donor rotations that vary role order, alternative counterfactual distributions, and within-template paired effect differences. Move from broad residual patches to head-output and path interventions, with necessity/sufficiency and faithfulness checks. Replicate on another model/backend before making general mechanistic claims.

## Quality verification and unresolved issues

The documented core command executed successfully. Final test outcome: **14 passed, 0 failed, 0 skipped**. Tests cover token/answer alignment, name-pool separation, counterbalancing, deterministic data/inference, no-ops, structural nulls, positive embedding repair, official-helper agreement, hook cleanup, metric sign, clustered resampling, file/figure provenance, development-only site selection and repeated pilot outputs. No computational blocker remains. A fresh-machine dependency installation and broad cross-platform reproducibility have not been tested.

The initial startup failed because unconstrained PyArrow 26 required NumPy 2 while TransformerLens 2.16.1 requires NumPy <2. Pinning datasets 2.21.0, PyArrow 17.0.0 and pandas 2.2.3 resolved the conflict; `pip check` passed. Python 3.11 was chosen directly rather than trying to force Python 3.14. A harmless upstream `torch_dtype` deprecation warning remains. One heatmap layout issue was fixed after visual inspection; raw experiment values were unchanged, and the complete core command was rerun. No publishing, repository push or submission was performed.

## Author contribution statement

The student commissioned and scoped this project. An autonomous AI assistant conducted the literature review, designed and implemented the experiment, executed it, produced the analysis and drafted these documents, with a research assistant agent reviewing primary sources. No independent student coding, interpretation, or mastery is asserted. The student should inspect the code, reproduce the experiment and demonstrate understanding before claiming personal technical contributions. The IOI task, causal-patching approach and original circuit findings belong to prior researchers. The local contribution is this reproducible implementation, explicit controls, and frozen-site name/template robustness evaluation; global novelty has not been established.

## References

- Wang, Variengien, Conmy, Shlegeris & Steinhardt (2022; ICLR 2023), [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593).
- Meng et al. (2022), [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262); [ROME project](https://rome.baulab.info/). Causal tracing precedent; this project does not perform weight editing.
- Zhang & Nanda (2023; ICLR 2024), [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042).
- TransformerLens [current patching documentation](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html), [4.0 migration/release notes](https://transformerlensorg.github.io/TransformerLens/content/news/release-4.0.html), and [executed v2.16.1 model source](https://github.com/TransformerLensOrg/TransformerLens/blob/v2.16.1/transformer_lens/HookedTransformer.py) / [patching source](https://github.com/TransformerLensOrg/TransformerLens/blob/v2.16.1/transformer_lens/patching.py).
- PyTorch [MPS backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).


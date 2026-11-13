# Protocol fixed before model results

Date: 13 November 2026 (Asia/Shanghai). Seed: 20261113.

Question: Can a clean residual-stream state restore the clean answer preference after a single repeated-subject name is changed, and does the selected final-position layer transfer to unseen names and templates?

Clean: `When Alice and Bob went to the store, Bob gave a book to` → Alice.
Counterfactual: same prefix, but `Alice gave ...` → Bob. The original clean answer remains the positive label in every metric. This is a grammatical counterfactual, not noise corruption. This is a conceptual replication of IOI causal interventions, not reproduction of the original full circuit or its exact dataset.

Design: 4 hand-written templates; 2 development and 2 held out. Two disjoint pools of 16 names, arranged in 8 disjoint unordered pairs per pool, with both name orders. Total planned n=128, 32 in each names × templates split. Only tokenization/alignment failures may be excluded; every candidate and reason is logged, including failures. No performance filtering. The pilot uses 4 development examples; it checks feasibility and control correctness, not effect-based tuning.

Patch `blocks.L.hook_resid_pre`, layers L=0..11, at BOS, first name, second name, repeated subject, final token. Zero-based layers denote input to that block. Cache only these residual sites. Fixed clean donor → counterfactual recipient. All patches replace one entire residual vector, not individual features. Prefix patches before the changed token and BOS are causal null controls. Repeat-subject layer-0 patch is a positive control (restores the only changed embedding). Final-position layer-0 is also a null control.

Discovery rule: among layers 6..11, select the layer with largest mean raw final-position patch gain on development names AND development templates. Freeze this selection and evaluate the other 3 splits, especially both held out. All-layer held-out plots are descriptive, not used for selection. No iterative hypothesis redesign after pilot.

Primary hypothesis: on both-held-out prompts, selected-layer final-position clean patches have positive mean raw logit-difference gain, and exceed unrelated-clean-donor patches. Hypothesis supported only if lower bounds of paired 95% cluster-bootstrap intervals exceed zero for BOTH comparisons. Otherwise report inconclusive/unsupported. This does not prove equivalence or nonexistence when unsupported.

LD = logit(clean answer) − logit(counterfactual answer). Gain = LD(patched) − LD(counterfactual). Secondary recovery = mean gain / mean(clean LD − counterfactual LD), only if mean gap >0.5; never clip and do not average unstable individual ratios. Report both pairwise name preference and full-vocabulary top-1 correctness. Pairwise preference is not full next-token accuracy.

Controls at every layer/final position: counterfactual self-cache no-op; clean self-cache no-op; unrelated clean donor rotated by two rows within the same template/pool (disjoint names, same semantic positions). No-op gains should be ≤1e-4 in magnitude. Wrong donor is a specificity diagnostic, not guaranteed null: it can carry generic task information. BOS and first-name/second-name patches are stronger structural nulls because these clean/counterfactual activations are identical by causality.

Uncertainty: 5,000 percentile bootstrap resamples of unordered name-pair clusters, preserving both orders and repeated templates. A cell has only 8 clusters; intervals describe sensitivity to these constructed pairs, not population-wide confidence. Do not treat 128 prompts or the layer sweep as independent replications. Templates are fixed and not randomly sampled; no inference to arbitrary language. Record seed, software, platform, model revision, hashes, runtimes and complete raw outputs. Validate MPS against CPU baseline and representative hooks; CPU fallback if unsupported or materially divergent (LD tolerance 0.02). Practical budget: one 124M model, float32, batch16, no training.

# Position-counterbalanced IOI extension: protocol v1

Fixed before any new model predictions. This is a prospective extension after observing the original study, not an independent replication of its hypothesis development. No new layer or position search is permitted.

## Fixed intervention and provenance

Use only `blocks.10.hook_resid_pre` at the final token for matched and unrelated-donor effects. This is zero-based block input, taken from the previous development-selected layer in `results/summary.json`; freeze a copy and SHA256 before running. Reuse the pinned GPT-2/tokenizer revision, environment and float32 implementation. Preserve all previous result files byte-for-byte; save this extension to a new output directory. Retain original README/report copies in `docs/history/position-counterbalanced-v1/`.

## Factorial prompts and held-out status

Eight disjoint evaluation name pairs (16 identities), eight disjoint donor pairs (16 more identities), two new templates, both introduction orders and both giver identities: **64 evaluation pairs and 64 donor pairs**. All identities are absent from the original 32-name dataset. Templates are fixed in source before predictions and differ from all original prompt strings. The two names in every pair switch giver/recipient roles independently of their introduction order. Within each identity × template, the identity is the clean answer once in first position and once in second position, and is giver equally often. Both clean-to-corrupted directions are included. These reverse-direction rows are dependent, not extra independent samples.

The task answer remains the recipient, as required by IOI. “Grammatical-role counterbalancing” means each identity occupies the giver and recipient roles equally; it does not mean recipient and giver are both correct answers to the same IOI question. Active voice and the syntactic recipient slot remain fixed. We do not claim all syntactic roles or sentence structures are balanced.

Templates:

1. `When {first} and {second} met at the station, {giver} gave a ticket to`
2. `Before leaving the library, {first} and {second} talked for a while. Then {giver} lent a book to`

Clean answer is the other (non-giver) identity. Corrupted changes only the repeated giver token to the clean answer; the correct counterfactual answer becomes the former giver. Preserve leading-space single-token answers, explicit BOS, exact name occurrence positions, continuation boundaries, equal clean/corrupted lengths, and exactly one changed token. Validate donor/recipient length and semantic-position alignment. Log every attempted candidate and any exclusions. A tokenization/alignment failure invalidates the entire evaluation+donor pair block, retaining balance; no replacement or outcome filtering. Abort if fewer than four complete blocks survive, preserving logs.

## Unmodified-baseline gate, before patching

Save all clean/counterfactual baseline measurements for evaluation AND donor prompts before computing experimental effects. Report pairwise success, full-vocabulary top-1, answer probabilities, mean LD and mean clean/counterfactual gap by template and clean-answer introduction position, separately for evaluation and donor pools. LD always equals logit(row's clean answer) minus logit(row's counterfactual answer), even when the clean answer is second.

Predeclare a usable stratum as clean pairwise accuracy ≥0.80, counterfactual pairwise accuracy ≥0.80, and mean LD gap >0.5. Overall restoration claims require ALL template × answer-position strata in BOTH pools to pass. Top-1 failures are always reported but do not gate the two-name LD metric. Preserve all baseline failures and all intervention rows; no performance-based exclusions. If a gate fails, report shifts and task failures, with no pooled normalized-recovery/restoration claim. Per-stratum normalized recovery is also suppressed for failing evaluation/donor strata. Baseline success does not by itself validate a circuit explanation.

## Donors and controls

Each evaluation name-pair block has an exclusive, disjoint donor-name pair (no donor sharing across blocks). Use the same template and introduction order. The matched donor is the evaluation row's own clean run. The unrelated-same-position donor has its clean correct answer in the same introduction position as the recipient's clean answer. The unrelated-opposite-position donor has its correct answer in the other position. Use both, report them separately, and average their per-example gains for a position-balanced unrelated comparison. Never describe either unrelated donor as a guaranteed zero-effect null.

At the frozen layer run four null/control patches: corrupted self-cache at final token; clean self-cache at final token; clean BOS into corrupted BOS; clean first-introduced-name state into its corrupted counterpart. The latter two precede the sole changed token and must be null by autoregressive causality. Tolerance for each no-op/null: absolute LD gain <1e-4. There is no layer sweep or layer-0 positive intervention in the extension. Validate the custom frozen-site hook against the official TransformerLens generic helper. Validate CPU/MPS LD at the SAME frozen site; tolerance 0.02, CPU fallback on failure.

Seven interventions per evaluation example, **448 total** if all eight blocks survive. Save per-example patched LD, raw gain, full-vocabulary top-1 and probabilities. Negative controls are excluded from restoration interpretation and retained as implementation checks.

## Hypothesis and uncertainty

Primary predeclared contrasts: (1) matched mean raw gain >0 in EACH clean-answer introduction-position stratum, and (2) matched minus per-example position-balanced unrelated mean gain >0 overall. Require positive lower endpoints of all three 95% cluster-bootstrap intervals AND the baseline gate to label the narrow restoration hypothesis supported; otherwise report unsupported/inconclusive. The two answer-position contrasts avoid allowing a strong effect in only one position to satisfy the hypothesis. These are conditional criteria, not proof of population-level generalization.

Secondary descriptive diagnostic: same-position minus opposite-position unrelated-donor gain; paired position differences and template subgroups. No post-result redesign or site selection. The earlier 88.3% unrelated-donor recovery used same-position donors only; compare it descriptively to the new same-position condition, not directly to the new balanced average as if datasets were identical.

Use seed 20261114 and 5,000 percentile bootstrap resamples of eight evaluation+exclusive-donor pair BLOCKS. Keep all orders, role reversals and templates together, and form paired contrasts before resampling. This accounts for fixed within-block donor dependence but not arbitrary donor-pool/assignment or template sampling. Report n and block counts. Raw gains are primary; ratio of mean gain to mean gap is secondary and unclipped, only when the baseline gate passes. No individual-ratio averaging. Fixed templates, eight constructed blocks, active voice, broad late residual replacement, fixed donor mapping, and one model limit conclusions.

## Execution and verification

Run tokenization/tests, baseline stage, then frozen-site patches. Rerun the full test suite. Repeat the extension without tuning and compare raw measurements. Save baseline failure cases, configuration, frozen prior-site hash, source hashes, original-result preservation audit, environment, device checks, raw CSV, cluster summaries, plots and execution logs. Update README/report with actual results and failures. Do not commit, publish or push.

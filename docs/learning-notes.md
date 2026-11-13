# Learning notes: understanding this experiment

## Transformer basics

A language model predicts a distribution over the next token from a sequence of previous tokens. A token is a vocabulary unit, not necessarily a word. GPT-2-small is a decoder-only autoregressive transformer with 12 blocks, a 768-dimensional residual stream, and 12 attention heads per block. Its approximately 124 million parameters remain fixed in this experiment. Inference does not train a model.

Tokens receive learned token and position embeddings. Blocks update the residual stream through attention and an MLP. A final layer normalization and unembedding project the final token's residual vector into vocabulary logits. Softmax converts logits to probabilities. A high rank among two candidate names need not imply the name is the highest-scoring vocabulary token.

## Attention mechanisms

Each attention head constructs queries, keys and values from residual states. Query-key scores, scaled and causally masked, produce attention weights through softmax. Weighted values are projected back into the residual stream. The causal mask prevents earlier tokens from attending to later tokens. An attention pattern by itself shows routing weights; it does not show the causal importance of the routed content.

In IOI, information about names and repetitions can be carried across positions. Wang et al. identify groups of heads involved in this behavior. Our residual patches include the combined effects of attention and MLP computations up to a site. We do not independently identify the head groups.

## Residual stream

Think of the residual stream as a sequence of vectors updated by each block: x ← x + attention(x), then x ← x + MLP(x), with the appropriate layer normalizations. `blocks.6.hook_resid_pre` is the state entering zero-based block 6, after blocks 0 through 5. It is not the output of block 6. All positions have vectors, but the final position determines this next-token prediction. Replacing a residual vector changes many potential features at once.

## Activation patching

Run the clean prompt and save activations. Run a paired counterfactual prompt, replacing one chosen vector with its clean counterpart. Measure the output change relative to the unmodified counterfactual. This is a causal intervention on the model's internal computation for these prompts. It asks whether the donor state can shift the recipient's output in this setting. It is not a proof that the patched state is necessary, contains a unique variable, or forms a complete circuit.

Our clean example is `When John and Mary went to the store, Mary gave a book to`, with John as the intended continuation. The counterfactual changes the second Mary to John, so Mary becomes the intended continuation. The sentences are both valid. The intervention tries to restore preference for the original clean answer; the label “corrupted” refers to the experimental direction.

## Logit differences

Let A be the clean answer and B the counterfactual answer. Every run uses LD = logit(A) − logit(B), with labels held fixed. Clean success is LD > 0; counterfactual success is LD < 0. Raw restoration gain is LD(patched) − LD(counterfactual). This avoids probability saturation, although it ignores all other vocabulary candidates.

Secondary normalized recovery is mean(gain) / mean(LD(clean) − LD(counterfactual)), provided the denominator exceeds 0.5. It is a ratio of means, not a mean of ratios. It can exceed 100% or be negative; neither is clipped. A 100% average LD recovery does not mean every original probability or internal computation has been recovered.

## Why controls matter

- Counterfactual self-cache patch: replacement with the very same vector should preserve output. A discrepancy suggests implementation or numerical trouble.
- Clean self-cache patch: the corresponding check in clean runs.
- BOS and first two name positions: these precede the only changed token, so causal attention makes clean and counterfactual states identical there. Their patches should have zero effect.
- Layer-0 subject patch: restoring the only changed token embedding reconstructs the clean input. This is a positive implementation check, not evidence of a discovered high-level mechanism.
- Unrelated clean donor: use the same template but a disjoint name pair. An effect may reflect generic task information, distributed features, or an unnatural hybrid. This donor is not a guaranteed zero-effect control. Compare matched and unrelated donors with paired intervals.
- Held-out names and templates: check whether a site selected on development data continues to work elsewhere. Selecting the best held-out layer and calling it transfer would leak evaluation information.

## Explanation of each experiment

1. **Tokenizer audit:** verify single-token leading-space answers, exact name occurrences, one changed token, equal lengths, and matched semantic positions. Exclusions are explicit; model errors are retained.
2. **Backend smoke test:** load GPT-2 on CPU, compare clean, counterfactual and one representative patched LD on MPS if available. Use float32. Availability alone is insufficient; unsupported operations or large discrepancies require CPU fallback.
3. **Pilot:** four development examples pass through the complete patch/control loop. Its job is feasibility and correctness. The fixed protocol does not change based on the effect sizes.
4. **Baseline:** evaluate clean and counterfactual next-token predictions. Save LD, pairwise success, full-vocabulary top-1 and answer probabilities.
5. **Residual sweep:** patch five semantic positions at twelve layers, preserving per-example values. Validate the custom hook against TransformerLens's official generic helper.
6. **Frozen-site transfer:** select one final-position layer among 6–11 on development names/templates only. Evaluate new names, new templates and both together. All-layer held-out plots remain descriptive.
7. **Uncertainty and controls:** resample unordered name-pair clusters, preserving both orientations and all repeated templates. A split contains 32 observations but only eight pair clusters. With a small constructed sample, the 95% intervals are sensitivity estimates, not assurance about arbitrary English.

## Technical interview questions and answers

**Why is activation patching stronger evidence than an attention map?** It directly changes a computation and measures a paired output effect. Attention weights alone do not show whether the values matter. The intervention remains conditional on prompts, site, metric and donor.

**Why can layer-0 subject recovery be trivial?** The counterfactual differs in exactly one token. Restoring that embedding with all other embeddings unchanged recreates the clean computation. It validates the machinery but does not locate a nontrivial downstream algorithm.

**Why not bootstrap 128 prompts independently?** Both name orders and multiple templates reuse the same pair. Treating those observations as independent overstates sample diversity. Cluster resampling respects that dependence. Only eight pairs per split still limits precision.

**Why use logit differences rather than probability recovery?** LD measures relative evidence on an unsaturated scale and can show negative effects. It is insensitive to a shared additive shift of the two logits but does not capture full-vocabulary behavior. We separately report top-1 accuracy.

**Does a positive patch show necessity?** No. It shows a donor state can shift a particular recipient computation. Necessity would require targeted removal or replacement controls, alternatives, and a suitable causal framework; even then redundancy matters.

**What does a negative held-out result mean?** The frozen intervention did not meet the predeclared criteria for this evaluation sample. It may reflect weaker task behavior, representation differences, donor mismatch, or insufficient statistical power. It does not refute the original IOI circuit result.

**How would you extend the project?** Expand name-pair diversity and templates before analysis; test alternative corruption distributions; use head output and path patching; compare necessity/sufficiency and circuit faithfulness; replicate on a second backend/model. Keep selection separate from evaluation.

## Attribution and study advice

Read the methods and raw examples before using this project in an application. The implementation and analysis were produced with substantial autonomous AI assistance; the student should claim only the work and understanding they personally acquire. The original IOI task and circuit are Wang et al.'s work; causal tracing and activation patching have earlier literature. Sources and executed results are in the research report.

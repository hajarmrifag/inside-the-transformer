# Inside the Transformer: Causal Tracing of Language Model Decisions

**Executed, AI-assisted research replication with a small robustness extension.** GPT-2-small; Apple Silicon; local inference; no training or paid API. Full provenance and limitations: [research report](docs/research-report.md).

## Question and motivation

Can replacing a counterfactual prompt's internal residual state with its clean counterpart restore the original answer preference, and does a development-selected site transfer to new names and templates? Causal explanations and their robustness matter to AI safety research. This small experiment is methodological practice, not a safety assurance result.

## Completed findings

128 prompts, four templates, two disjoint name pools; **zero exclusions**. Both name orders are used. The experiment completed 256 clean/counterfactual baseline evaluations and **12,288 interventions**, with a fixed seed and raw per-example outputs. Final-token input to zero-based block **10** was selected only on development examples.

| Evaluation split | Prompts / pairs | Clean LD | Counterfactual LD | Patch gain [95% CI] | Gap recovery |
|---|---:|---:|---:|---:|---:|
| Development | 32 / 8 | 4.141 | -4.325 | 8.627 [7.423, 9.762] | 101.90% |
| New names | 32 / 8 | 4.305 | -5.012 | 9.636 [8.947, 10.316] | 103.42% |
| New templates | 32 / 8 | 4.031 | -3.256 | 7.091 [6.170, 7.966] | 97.30% |
| Both held out | 32 / 8 | 4.371 | -3.601 | 7.993 [7.334, 8.649] | 100.27% |

CI = 95% cluster-bootstrap interval, with eight unordered name pairs per cell; 5,000 resamples. Recovery is a ratio of aggregate means, can exceed 100%, and has no interval here. Raw gain is primary. All unmodified clean and counterfactual prompts had correct pairwise and full-vocabulary top-1 predictions. Matched patches on both-held-out prompts restored the clean pairwise preference in 100% of examples; patched full-vocabulary top-1 was not measured.

**Important qualification:** unrelated clean donors recovered 88.3% of the held-out gap. Matched donors exceeded them by 0.950 [0.590, 1.292] LD units, but most restoration was not donor-name-specific. Clean answers always occupy the first introduced-name position. This supports a narrow transfer claim, not discovery of a complete computational circuit.

## Methods and validation

Clean ABB prompts versus grammatical ABA counterfactuals differ at one repeated subject token. Answer continuations are verified single GPT-2 tokens. LD always means clean-answer logit minus counterfactual-answer logit. Patch one whole residual vector at 12 layers × five semantic positions. Self-cache, pre-change/BOS and unrelated-donor controls are included; layer-0 subject repair is a positive implementation control. [Protocol](docs/protocol.md) was written before results.

Model and tokenizer revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; Python 3.11.17; TransformerLens 2.16.1; PyTorch 2.15.0; float32. Verified MPS/CPU LD discrepancy: 0.00001335. Official/custom patch discrepancy: zero. No-op/null/embedding-repair errors: zero. Two pilots and two complete runs matched numerically. **14 tests passed; zero failed or skipped.** The final experiment took 30.7s with cached weights; initial download/setup takes longer.

## Visualizations

![Final-token effects across layers](results/figures/recovery_layers.png)

![Held-out controls](results/figures/controls.png)

Also see [baselines](results/figures/baseline.png), [patching heatmap](results/figures/patching_heatmap.png), and [template comparison](results/figures/templates.png). All five are in the [full report](docs/research-report.md), with real measurements and uncertainty labels.

## Reproduction

From this project directory, create the environment once (Python 3.11 preferred; 3.12 supported by the declared range but not tested here):

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

Run the complete core experiment, aggregation, five figures and tests with **one command**:

```sh
sh scripts/reproduce.sh
```

By default this writes `results/reproduced/`, preserving the delivered results. To select an output folder use `sh scripts/reproduce.sh results/my-run`. The executed delivery used `HF_HUB_OFFLINE=1 sh scripts/reproduce.sh results` after weights were downloaded. Internet is needed for dependency/model download; cached runs can set `HF_HUB_OFFLINE=1`. No login or paid service is required. The runner validates MPS and falls back to CPU. A manual CPU experiment can use `.venv/bin/python -m inside_transformer.experiment --device cpu --output results/cpu`, followed by the analyze module on that folder.

Tests run on CPU and use cached weights. Some artifact tests inspect the delivered `results/`; fresh checkout artifact tests skip until a run is present. The delivery has no skips. Model weights stay in the Hugging Face cache, not in this folder. The lock was executed on macOS arm64; a fresh-machine install and other platforms were not tested. Full raw-file repeat comparisons are recorded in `results/full_reproducibility.json`. These are same-machine checks, not cross-hardware exactness guarantees.

## Files and learning

- [Exact file inventory](FILES.md)
- [Research report](docs/research-report.md), [learning notes and interview answers](docs/learning-notes.md), [protocol](docs/protocol.md), [application summary](docs/application-summary.md)
- `src/inside_transformer/`: deterministic data, patch hooks/metrics, experiment and analysis
- `tests/`: alignment, interventions, metrics, reproducibility and artifact checks
- `results/`: raw `baselines.csv` / `interventions.csv`, dataset/exclusions, aggregates, figures, configuration, metadata, hashes, pilot outputs and preserved first full run
- `requirements-lock.txt` / `pyproject.toml`: executed frozen dependencies / project constraints
- `research_notes/` and `reports/`: primary-source research and pre-results background memo

## Limitations

Four fixed templates; eight name-pair clusters per cell; no answer-position balancing across clean/counterfactual roles; broad residual replacement; a strong unrelated-donor effect. Late-state recovery does not identify a circuit or establish necessity. Bootstrap intervals are conditional on the small constructed sample. No training-data audit, second model, demographic study, path patching, or cross-machine replication was performed. The original IOI paper's full circuit was not replicated.

## Author contribution

The student commissioned and scoped this project. An autonomous AI assistant conducted the literature review, designed and implemented the experiment, executed it, produced the analysis and drafted these documents, with a research assistant agent reviewing primary sources. No independent student coding, interpretation, or mastery is asserted. The student should inspect the code, reproduce the experiment and demonstrate understanding before claiming personal technical contributions. The IOI task, causal-patching approach and original circuit findings belong to prior researchers. The local contribution is this reproducible implementation, explicit controls, and frozen-site name/template robustness evaluation; global novelty has not been established.

## References

- Wang, Variengien, Conmy, Shlegeris & Steinhardt (2022; ICLR 2023), [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593).
- Meng et al. (2022), [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262); [ROME project](https://rome.baulab.info/). Causal tracing precedent; this project does not perform weight editing.
- Zhang & Nanda (2023; ICLR 2024), [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042).
- TransformerLens [current patching documentation](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html), [4.0 migration/release notes](https://transformerlensorg.github.io/TransformerLens/content/news/release-4.0.html), and [executed v2.16.1 model source](https://github.com/TransformerLensOrg/TransformerLens/blob/v2.16.1/transformer_lens/HookedTransformer.py) / [patching source](https://github.com/TransformerLensOrg/TransformerLens/blob/v2.16.1/transformer_lens/patching.py).
- PyTorch [MPS backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).


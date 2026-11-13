# Reproduction

## Tested scope

Recorded runs used Python 3.11.17, macOS 26.5 arm64, PyTorch 2.15.0, TransformerLens 2.16.1 and float32. MPS was compared with CPU at representative baseline/intervention sites before use. CPU is the fallback. Package constraints admit Python 3.12, but it has not been tested here. The lock is a frozen environment listing, not a cross-platform or hash-locked dependency guarantee.

Same-machine repetition and execution of this release candidate are checked in the publication audit. No fresh-machine installation, new empty-model-cache installation, cross-hardware replication or second model is claimed.

## Setup

Run from the repository root. Creating this environment is a reproduction instruction; the audit uses the existing tested environment and does not claim to have validated this setup on a fresh machine.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python scripts/cache_model.py
.venv/bin/python -m pip check
```

Editable installation uses the declared isolated build requirements; disabling build isolation would require providing the build tools yourself.

The last model-setup command downloads the pinned public GPT-2 weights and tokenizer once. It needs internet, but no paid service, training, account or API key. Model/tokenizer revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Weights remain in the Hugging Face cache outside the repository. Model acquisition and a clean-cache download were not repeated during the audit; a cached same-machine call was checked. The extension deliberately uses cached files and forces Hugging Face offline mode, so this setup step is required before its first run.

The pinned TransformerLens 2.16.1 uses `HookedTransformer`; current 4.x documentation describes a different loading API. Do not upgrade TransformerLens independently of the environment lock. Initial dependency resolution required pinning PyArrow 17.0.0 and datasets 2.21.0 because newer PyArrow conflicted with NumPy <2.

## Core commands

```sh
sh scripts/reproduce.sh
sh scripts/reproduce_counterbalanced.sh
```

Defaults write to `results/reproduced/` and `results/counterbalanced-reproduced/`. Each command refuses an existing destination; supply a new folder to repeat safely:

```sh
sh scripts/reproduce.sh results/reproduced-check-2
sh scripts/reproduce_counterbalanced.sh results/counterbalanced-reproduced-check-2
```

Both commands run the experiment, aggregation, figures, an independent numeric audit of the newly generated results, and the test suite. Artifact tests also validate the bundled reference datasets. This distinction matters: a passing reference-artifact test alone does not verify a newly requested output directory. The numeric audit now explicitly receives that directory.

For fully cached original runs, set `HF_HUB_OFFLINE=1`. Override `PYTHON` if using another environment. For explicit CPU inference, invoke the experiment module with `--device cpu`, then its analysis module. The extension selects its backend only after a same-site CPU/MPS comparison; it does not search new intervention sites.

## Audit without new inference

```sh
.venv/bin/python scripts/audit_results.py
.venv/bin/python -m pytest -q
```

The independent audit recomputes summary and aggregate statistics from CSV files without importing the project's analysis or metric implementations. It checks paired contrasts, fixed-seed cluster-bootstrap intervals, counts, donor mappings, failures and hashes. It does not prove that arbitrary CSV files came from a model. Recorded runs, executable hook tests and same-machine inference repetitions are separate evidence.

The preserved result records include same-machine repeats. Runtime and metadata files differ between executions; numerical-repeat claims refer to the compared measurements, not byte identity of every file. Historical result metadata hashes describe the source snapshot at that run, not every later documentation or packaging change.

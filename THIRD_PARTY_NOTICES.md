# License scope and third-party notices

## Original project materials

The root [LICENSE](LICENSE) is the standard MIT License, copyright 2026 Hajar Mrifag. It covers original project software and its associated documentation, including `src/`, `scripts/`, `tests/` and the recovered original plotting source under `audit/historical-source/`. AI assistance is disclosed in [docs/ai-assistance.md](docs/ai-assistance.md); attribution to the project owner does not imply independent student implementation or ownership of third-party work.

The license does not replace the terms for external libraries, model/tokenizer assets, cited literature, or the upstream notices in `licenses/`. The synthetic prompt records, numeric measurements and project-generated figures are retained as original research evidence. No separate data/content license is added by this code-license selection, and no rights in external training data are asserted.

## Software used, not bundled

This publication file set contains dependency declarations and API calls, not installed dependency packages, vendored third-party modules, font files or model weights. Review found no substantial copied third-party implementation or downloaded benchmark dataset in the original source. Hooking conventions and the IOI method are credited rather than represented as original inventions. This is a scoped source/asset review, not an exhaustive external code-similarity audit.

| Component | Version used | License evidence and retained obligations |
|---|---|---|
| TransformerLens | 2.16.1 | MIT; copyright 2022 TransformerLensOrg. The [original notice](licenses/TransformerLens-MIT.txt) is preserved verbatim and matches the installed notice. [Official tagged source](https://github.com/TransformerLensOrg/TransformerLens/blob/v2.16.1/LICENSE). Preserve its notice when distributing copies or substantial portions. |
| Hugging Face Transformers | 4.57.6 | Apache-2.0, as declared in installed distribution metadata and license files. It is installed separately and is not relicensed by this repository. |
| Hugging Face Datasets / Apache Arrow | 2.21.0 / 17.0.0 | Apache-2.0 package metadata/notices; no dataset download is used by this experiment. Packages are not bundled. |
| PyTorch | 2.15.0 | The installed wheel declares multiple component licenses, including BSD, Apache, MIT and others. Its full declared expression and license-file hashes are recorded in the dependency inventory; calling the whole wheel simply MIT or BSD would be inaccurate. |
| NumPy / pandas | 1.26.4 / 2.2.3 | BSD-family notices, with bundled-component notices retained in the installed packages. No binary distribution is included here. |
| Matplotlib | 3.11.2 | Matplotlib/PSF-based license and bundled notices. Only project-generated raster plots are included, not Matplotlib or font software. |
| pytest | 9.2.0 | MIT; separately installed, not vendored. |

The [dependency license inventory](audit/dependency-licenses.json) records all packages in the frozen requirements file, their installed versions, declared license metadata and license/notice hashes. It is an environment inventory, not a claim that every dependency is MIT or that redistributing their binaries would have no further obligations. If later bundling wheels, source, fonts or containers, review each component's redistribution, notice and modification requirements and preserve its actual notices; this MIT file does not waive them. Existing installed notices were not changed.

## GPT-2 model and tokenizer

The model/tokenizer are downloaded separately at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` and are absent from the publication file set. The [pinned Hugging Face model card](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/README.md) declares MIT. OpenAI's repository carries a **Modified MIT License**, copyright 2019 OpenAI; its [unmodified upstream notice](licenses/GPT-2-Modified-MIT.txt) is retained, including its output/responsible-use language. This is distinct from Hajar Mrifag's standard project MIT License. [Official OpenAI notice](https://github.com/openai/gpt-2/blob/master/LICENSE).

Model-based predictions, logits and measurements in `results/` were produced with GPT-2-small as explicitly described in the README/report. They are not a copy of OpenAI's WebText training corpus. The project does not claim that prompts are absent from pretraining, own the pretrained model, or relicense its underlying training data. If redistributing model/tokenizer files later, include and review their applicable upstream terms rather than assuming the project MIT notice is sufficient.

## Research literature and datasets

The IOI task and prior circuit/token-position analysis belong to [Wang et al.](https://arxiv.org/abs/2211.00593). Causal tracing and patching-methodology precedents are cited in the research report and literature audit. This project supplies hand-authored task-like prompts and its executed measurements, not a redistributed published benchmark or paper figures. Paper text, figures and external datasets are not placed under this project's MIT grant merely by being cited.

Upstream notice download URLs and content hashes are recorded in [audit/third-party-notice-sources.json](audit/third-party-notice-sources.json). Their original rightsholder names and license wording are preserved.

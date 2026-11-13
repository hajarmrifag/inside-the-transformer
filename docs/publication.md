# GitHub release preparation

## Suggested repository metadata

Name: `inside-the-transformer`

Description: **AI-assisted GPT-2-small activation-patching replication with position-counterbalanced IOI prompts, paired donor controls, failure analysis and reproducible local experiments.**

Topics: `mechanistic-interpretability`, `activation-patching`, `gpt2`, `transformerlens`, `ai-safety`, `reproducible-research`, `pytorch`, `causal-interventions`.

## Recommended layout

```text
README.md
LICENSE
THIRD_PARTY_NOTICES.md
licenses/                 # unchanged upstream notices
pyproject.toml
requirements-lock.txt
.gitignore
docs/
  research-report.md
  position-counterbalanced-results.md
  protocol.md
  position-counterbalanced-protocol.md
  reproduction.md
  learning-notes.md
  ai-assistance.md
  application-summary.md
  publication.md
  history/                 # frozen prose and preservation hash inventory
src/inside_transformer/    # original and counterbalanced implementations
scripts/                  # caching, reproduction and independent numeric audit
tests/                    # model, alignment, metric, artifact and tamper tests
results/                  # small raw data, repeats, figures and run metadata
audit/                    # audit findings and numeric claim ledger
FILES.md
```

The staged candidate follows this layout. Small raw repeats are retained because they substantiate numerical repeat claims. Virtual environments, caches, package build output, exploratory research notes, redundant pre-results memos, legacy prose generators and two path-bearing extension logs are excluded. Canonical measurements, configurations, raw failures, figures, protocols and hash records are retained. A release-specific manifest records omissions; original local records are preserved.

## License and publication authorization

The owner has explicitly selected MIT for original project code and authorized preparation of a clean local initial commit. Creating a remote, publishing, pushing or submitting remains unauthorized until explicit approval is provided. The final local commit hash is reported after committing rather than embedded in its own files.

**Original project software and associated documentation: MIT, copyright 2026 Hajar Mrifag.** The author name matches the existing Git identity; no email address is added to the LICENSE. See the root LICENSE and THIRD_PARTY_NOTICES.md for scope. Dependencies, GPT-2 assets and literature retain their upstream terms. No separate data/content license is asserted, and no institutional affiliation, remote URL or GitHub account has been invented. The current Git commit uses the configured author identity; it has not been published.

Fresh-machine and cross-hardware validation are unperformed limitations, not claimed results. The substantive scientific limitations remain even if the owner approves publication: eight constructed blocks, two templates, open-vocabulary baseline failures, whole-vector late residual patches, and confounded donor position/role.

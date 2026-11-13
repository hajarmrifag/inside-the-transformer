# Final research-integrity and release audit

**Historical status note:** This report records the pre-license, pre-commit review. The owner has since selected MIT and authorized a local initial commit; see [publication-readiness.json](publication-readiness.json) and [third-party notices](../THIRD_PARTY_NOTICES.md). Its former license/commit blockers below describe that earlier phase. Public remote creation and pushing still require explicit approval.

Audit date: 13 November 2026. Scope: application-portfolio release of the existing original study and counterbalanced extension. No new research extension, site tuning, dataset selection or outcome filtering was performed.

## Assessment

The arithmetic and reruns agree with the recorded measurements. No numerical discrepancy or undisclosed performance-based exclusion was found. The defensible claims concern conditional intervention effects; name-specific mechanistic isolation, complete-circuit discovery and population-level significance are not established. The candidate is ready for owner review as an AI-assisted research portfolio artifact. It is not an externally peer-reviewed novel research finding, and is not approved for publication or submission.

**Remaining publication decision: the owner has not selected a license.** Do not label the repository permissively licensed open source until that decision is made. Explicit approval is still required before committing, creating a remote, publishing, pushing or submitting. Fresh-machine, empty-cache setup and cross-hardware replication remain unperformed validation limits rather than implied accomplishments.

## Materials inspected

Reviewed all eight scientific source modules, reproduction and report-generation scripts, tests, original and extension protocols, README, research reports, learning/application notes, raw datasets/baselines/interventions, all summaries/aggregates, failure files, verification records, repeated runs and cited primary literature. Earlier figure previews were checked; the audit reran the existing figure-generation commands. The source and canonical measurement records were not altered by this audit. Documentation and reproduction/audit tooling were improved.

## Independent numerical checks

The new [numeric audit](../scripts/audit_results.py) recomputes **2,551 fields** from raw measurements without importing project metric or analysis functions. All matched within the declared numerical tolerance. Its [JSON result](numeric-audit.json) and [CSV claim ledger](numeric-audit.csv) include counts, means, cluster-bootstrap endpoints, accuracy, recovery ratios, all aggregate rows and paired contrasts. Twenty-nine artifact hashes were checked as part of this pass.

| Evidence | Verified facts |
|---|---|
| Original study | 128 pairs; 256 unmodified evaluations; 12,288 interventions; four templates; zero exclusions; all baseline top-1 correct. Each cell has 32 pairs and eight reused name-pair clusters. |
| Original both-held-out cell | Matched gain 7.993 [7.334, 8.649]; paired matched-minus-unrelated 0.950 [0.590, 1.292]; unrelated recovery 88.3%. |
| Counterbalanced extension | 64 evaluation plus 64 donor directed pairs; 448 interventions; two templates; eight exclusive recipient+donor blocks; zero exclusions. Reverse directions reuse sentences. |
| Extension baseline | Evaluation clean/counterfactual pairwise 62/64 = 96.875%; top-1 42/64 = 65.625%. Donors have 100% pairwise but 62.5% top-1. All baseline gate strata pass the specified two-name criteria. |
| Retained failures | Two distinct evaluation clean name-preference errors, 22 evaluation and 24 donor clean top-1 errors (all ` the`), and six matched-patch name-preference failures. Counterfactual reversals are not additional independent sentences. |
| Extension interventions | Matched 6.091 [5.441, 6.886]; unrelated same-position 5.575 [5.260, 5.866]; unrelated opposite-position 0.368 [−0.103, 0.965]. |
| Proper paired comparisons | Same-minus-opposite donor 5.207 [4.641, 5.704]; matched-minus-balanced donor 3.120 [2.771, 3.554]; matched-minus-same donor 0.516 [0.0012, 1.197]. |
| Controls | Original no-op/prefix nulls and extension four null conditions have zero gain; the original embedding repair exactly reconstructs clean LD. Helper comparisons and sampled CPU/MPS agreement are recorded separately. |

The stored confidence intervals use 5,000 seed-fixed percentile resamples of equal-size cluster means. Paired differences are formed before resampling. They are not inferred from overlap/non-overlap of separate condition error bars. The near-zero lower endpoint for matched-minus-same-position is fragile, not compelling evidence for name-specific isolation. There are no p-values and no population-level or simultaneous-coverage guarantees. The original cyclic donor assignment also induces cross-pair dependence, limiting its interpretation further.

The auditor is itself AI-assisted; “independent” here means a separate arithmetic implementation, not an independent human review or evidence that arbitrary CSVs came from a model. Hashes establish content consistency, not independent chronology or authenticity. Executable tests and inference repetition provide separate support.

The latest canonical original and extension source hashes match the current scientific files. Earlier pilot and first-run records contain an older `analyze.py` hash. The audit reconstructed that version by reversing the documented heatmap-layout change and verified an exact SHA256 match against all three older records. The [recovered plotting source](historical-source/analyze-before-layout-fix.py) and [recovery record](source-recovery.json) are bundled; old hashes were not rewritten. The other recorded scientific files match current source. This restores content availability, not independent evidence of historical execution or a Git history that did not exist. See [source-history check](source-history.json).

## Re-execution and tests

Executed the original and extension reproduction commands from an isolated candidate source directory, using the **existing same-machine environment and cached model**. Imports were checked to come from candidate source. The original command regenerated 128 baseline rows and 12,288 intervention rows; the extension regenerated 128 evaluation/donor baseline rows and 448 intervention rows. Both commands ran their analysis/figures, explicitly audited the newly requested output directory, and passed **24 tests, zero failures and zero skips**.

Compared raw baseline and intervention measurements against the bundled references: maximum numeric difference was **0.0** for both experiments. CSV byte hashes also matched. See [re-execution evidence](reexecution.json), [original command log](original-command.log) and [extension command log](extension-command.log). Logs here use logical roots instead of personal paths. The cached-model setup command was executed successfully offline; this does not test an empty-cache download. New model runtimes were approximately 29.4 seconds and 3.6 seconds, excluding process startup/analysis/tests.

The suite includes the original 14 tests, eight extension tests, and two new independent-audit tests. A tamper test deliberately changes the paired contrast's lower interval endpoint and confirms that the audit rejects it. A passing reference-data test alone is not offered as proof that a new output was checked: the reproduction commands now pass the new directory to the numeric auditor. Package consistency was also checked with `pip check`.

## Documentation and attribution changes

- Replaced a long, duplicated README/report narrative with a clear separation of original study and extension, prominent baseline failures, correct units and sample counts, paired contrasts, and a linked detailed record.
- Distinguished logit-gap recovery, pairwise preference and full-vocabulary top-1. Reported matched top-1 is not extrapolated to the original study, which did not save it.
- Clarified that protocol records are local, not externally preregistered. The original sweep was collected before automated development-only selection; the extension site was already fixed.
- Added explicit cached-model setup, preserved-output safeguards, output-specific numeric verification and precise same-machine validation scope.
- Verified primary references, including Wang et al.'s prior token/position experiments. ROME is a precedent, not claimed as the unique origin of causal interventions. No paper result is mislabeled as a local measurement. See [literature audit](literature-audit.md).
- Disclosed extensive AI assistance in design, coding, review, execution, analysis, writing and audit. No independent student implementation or demonstrated mastery is invented. The 70-word application description and one-line CV wording state AI assistance.

## Privacy, size and Git staging

The source review found two extension runtime-log lines containing personal absolute output paths. Original local logs remain preserved; those two logs are omitted from the public candidate and identified by relative path/hash in `release-exclusions.json`. No canonical measurement, failure, figure or summary was redacted. All **44 original result files remain byte-identical**, and extension canonical files are unchanged apart from omission of the private log. See [preservation evidence](preservation.json).

The candidate excludes the approximately 1.3 GiB virtual environment, caches, generated package metadata, redundant working research memos and legacy prose generators. Small raw repeated measurements are intentionally retained to support numerical-repeat claims. The staged payload is approximately 4.4 MB; the largest file is under 1 MB. No weights, oversized model files or detected credential formats are included. The [staged-file scan](release-scan.json) is heuristic, not a guarantee against every possible private datum or secret. New scratch reproduction outputs are ignored and excluded from the release payload.

A local Git index stages the candidate without a commit or remote. The release archive is assembled from the index export, so ignored working-directory artifacts cannot accidentally enter it. The working research directory and its original results remain separate. Current release files are listed in `FILES.md`; historical Markdown copies retain their original repository-root link context and older verification counts, and are not the current reproduction guide.

## Limits that remain

Eight constructed blocks, two fixed templates, limited donor mapping, substantial station-template top-1 failures, fixed active voice/recipient slot, broad late residual replacement, and coupled donor position/role. No unique feature, necessity, complete circuit, model safety or global novelty follows. No fresh-machine installation, cross-hardware/model repetition, human ambiguity review, demographic study, training-data contamination check or external peer review was performed. These limits are documented, not treated as missing evidence that may be assumed favorable.

Suggested GitHub description, topics and layout are in [publication notes](../docs/publication.md). No publication action is authorized or performed by this audit.

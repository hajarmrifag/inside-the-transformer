# Primary-source attribution audit

Checked on 13 November 2026. These references support background/method attribution; project effect sizes come from its own CSV files.

| Reference | Checked claim and boundary |
|---|---|
| Wang, Variengien, Conmy, Shlegeris & Steinhardt, [Interpretability in the Wild](https://arxiv.org/abs/2211.00593), 2022 preprint / ICLR 2023 | Established GPT-2-small IOI and a head-level causal circuit analysis. The [v1 text](https://arxiv.org/html/2211.00593v1) reports 26 heads in seven classes. This project does not reproduce the full circuit or its faithfulness/completeness/minimality evaluation. |
| Wang et al., [v1 Appendix A](https://arxiv.org/html/2211.00593v1#A1) | Already investigates token and positional signals using altered-name and altered-position donors at S-Inhibition heads. The new whole-residual donor-position comparison is a local robustness exercise, not discovery of position signals or a replication of that appendix's precise experiment. |
| Meng, Bau, Andonian & Belinkov, [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262), NeurIPS 2022 | Causal-tracing precedent plus ROME weight editing. This project performs activation replacement, not ROME editing. It does not claim Meng et al. uniquely originated all activation-patching methods. |
| Zhang & Nanda, [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042), 2023 preprint / ICLR 2024 | Metrics and corruption choices can materially affect localization conclusions. Their recommendations motivate explicit metric/corruption reporting, not universal superiority of this project's design. |
| TransformerLens [v2.16.1 patching source](https://github.com/TransformerLensOrg/TransformerLens/blob/v2.16.1/transformer_lens/patching.py) | Confirms generic patch helper and position-replacement implementation used to validate custom hooks. The stored helper checks compare the selected metric at a representative site, not every possible intervention. |
| TransformerLens [4.0 release](https://transformerlensorg.github.io/TransformerLens/content/news/release-4.0.html) | Confirms removal of `HookedTransformer`. The executed project intentionally remains pinned to 2.16.1; current documentation is not its executable API specification. |

No numerical values from the papers are presented as this project's measurements. No quoted passages, paper figures, model weights or vendored library source are included. The review is focused, not an exhaustive novelty search or external peer review.

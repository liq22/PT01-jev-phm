# Closest Sources and Transfer Decisions

## Source hierarchy

### Review scope: 23 September 2026

The discovery entry was [awesome-jev-zh, especially Cold Perspective](https://github.com/yzfly/awesome-jev-zh#-冷静看待). It is a community index, not the source of model-performance evidence. This review followed its official guidance and the original community experiments. No Jev API calls, community benchmark reruns, or PHM training were performed. Star counts and promotional speed ratios are not evidence of scientific novelty.

| Source | Material actually inspected | Consequence for this paper |
|---|---|---|
| [TypeSafe launch, 2026-09-15](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Official search-indexed article including evaluation caveats; direct page open failed | Transfer state-to-typed-decision interface. Do not inherit RLCD, latency ratios, or calibration guarantees. Vendor workflow references include other models' probabilities rather than an independent PHM ground truth. |
| [Official TypeSafe skill](https://github.com/typesafe-ai/skills/blob/65a39f393687675ce170e6094757de20370365b9/skills/typesafe-ai/SKILL.md) | Complete official GitHub document | Code owns calculations and composition. Independent questions cannot observe one another's answers. Noul is event probability without separate confidence; Choice/Score confidence is not workflow correctness. Named question IDs alone do not convey meaning to the model. |
| TypeSafe live confidence, primitives, composition, and jaggedness pages | Both normal and Markdown URL retrieval attempted; unavailable | Do not claim to have inspected current endpoint schemas or the detailed failure catalog. Official GitHub guidance supports design semantics, not an implemented live API integration. |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | Complete README and docs/ATOMIC_PLANNING.md | Qwen3-0.6B, direct heads, question/candidate conditioning, and local judgments with code composition already exist. Shared-prefix inference is listed as future work; ordinary batching is not the proprietary sampler. Its code-guided maze example separates a planner's success from the model's own measured contribution. |
| [Phishing benchmark](https://github.com/anisselbd/jev-phishing-bench) | Complete README including original task, post-review controls, question provenance and limitations | Treat atomic decomposition plus supervised fusion as an empirical design, not a pure model-only effect. Match downstream supervision, available evidence, question count, and traditional controls. |
| [Spam evaluation](https://github.com/bitnovus/jev-spam-eval) | Updated README through experiments and limitations | Context enrichment, category definitions and development feedback materially affect results. The latest source is more qualified than the index's statement that distribution robustness is a stable advantage. Restore evidence before crediting architecture. |
| [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) | Official model card; technical-report metadata | Retain the selected checkpoint. Actual local weight revision and inference remain separate execution prerequisites. |
| [Goren et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c8b100b376a7b338c84801b699935098-Abstract-Conference.html) | Publisher record and original arXiv:2405.11533v2 HTML, especially setup, inference rules, threshold assumptions and experiments | Calibrated leaf probabilities, hierarchy sums, partial rejection and risk-coverage methods are prior art. No automatic inheritance of their exchangeability-based guarantee. |
| [Valmadre, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/727855c31df8821fd18d41c23daebf10-Abstract.html) | Publisher abstract/metadata; attempted arXiv HTML unavailable | Author reports that flat softmax can dominate top-down classifiers. Keep a strong flat control; no detailed implementation equivalence is inferred from the abstract. |
| [Koh et al., ICML 2020](https://proceedings.mlr.press/v119/koh20a.html) | Publisher abstract/metadata | Concept-to-label prediction is prior art. Our label-derived fault-presence/identity nodes are not independently validated physical concepts. |
| [Gneiting and Raftery, JASA 2007](https://doi.org/10.1198/016214506000001437) | Publisher metadata and abstract | Proper-scoring foundations are existing theory. Supervised CE/Brier is not evidence that proprietary RLCD has been reproduced. |

The prior revision's source conclusions for FaultGPT, CLSGen, CAT, Guo temperature scaling, and Geifman selective classification are retained. CLSGen metadata was rechecked this review. The other inherited works were not all reread in full; their previously recorded scope remains: FaultGPT and CLSGen original method HTML; CAT publisher-indexed method text; Guo and Geifman publisher/original abstract and metadata. No new absence claim depends on an unread full text.

## What the original experiments actually allow

### Decomposition is a workflow contrast

The phishing source reports 62.6% direct-verdict accuracy on all 2,000 examples. Its post-review held-out half-B comparison reports 95.0% for regression over five Jev signals, 91.8% for the regex-based comparison, and 93.2% for regression over matched Haiku signals, with the last accuracy comparison p=0.063. These numbers have different fitting/evaluation contexts; subtracting 62.6 from 95.0 is not a clean estimate of decomposition alone. Failure to reject a difference is not proof of equivalence. The signal questions were informed by the dataset's URL taxonomy, and email bodies were synthetic. These are author-reported results, not independently reproduced here.

The updated spam source reports 93.62% to 97.98% main-set accuracy when the question is held fixed and omitted evidence is restored. On its 5,733-example main set, evidence-focused Jev is reported at 98.64% versus 98.87% for matched-evidence TF-IDF regression. It labels development as exploratory, notes unknown pretraining exposure and duplicate/source concerns, and limits its later phishing-only set to recall rather than deployable precision. The older 97.3% versus 72.5% temporal example remains one restricted collection, not a general PHM or distribution-shift guarantee.

## Decisions absorbed into the manuscript

### Keep, change, and defer

Keep the existing PHMFactory/Qwen direct probe and E0/E1 protocol. Keep unit-weighted calibration as a secondary, population-specific hypothesis. Change the paper's framing so it does not mistake that weighting for Jev's central mechanism.

Specify one minimal extension in main Section 4.4: independent fault-presence and hypothetical conditional fault-identity questions, composed by the probability chain rule. Existing labels provide these targets when the taxonomy permits them. Do not claim physical-symptom interpretation, arbitrary dynamic candidates, a new hierarchy algorithm, or automatic calibration.

The necessary next design contrast is flat versus factorized prediction with numerical and Qwen controls, plus a same-question-count direct control. State enrichment and larger-model/generative comparisons remain conditional on a concrete evidence or efficiency claim. No model weights or source implementation are imported from community repositories in this revision.

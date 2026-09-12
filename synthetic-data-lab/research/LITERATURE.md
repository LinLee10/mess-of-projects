# Literature coverage and implementation boundaries

This ledger accounts for the thirty paper records in the earlier research extension. It separates literature relevance from software execution. The earlier archive contains fuller mechanism notes and records unequal reading depth: method sections for selected papers, abstract screening for others, and a failed full text retrieval for the Nature collapse paper. This file does not upgrade those records into claims that every paper was reread completely during the second review.

The second review checked the defining CTGAN and numerical diffusion mechanisms against their primary papers, reconsidered the relationship between the language and tabular tracks, and checked selected primary metadata and recent candidate abstracts. No exhaustive systematic survey or complete current frontier comparison is claimed. Published paper scores are not local measurements.

## Original language model program

| Record and primary source | Role and execution boundary |
| :--- | :--- |
| [Self Instruct](https://arxiv.org/abs/2212.10560) | Seed instruction bootstrapping. Relevant to the upstream text program; no pretrained teacher or student run in this extension. |
| [WizardLM and Evol Instruct](https://arxiv.org/abs/2304.12244) | Instruction evolution. The text program must preserve response regeneration and evolution selection; not executed here. |
| [Magpie](https://arxiv.org/abs/2406.08464) | Raw unfinished user prefix sampling, not ordinary question generation prompting. Not executed here. |
| [STaR](https://arxiv.org/abs/2203.14465) | Answer checked rationales and iterative training. A single rationale generation pass is not the complete method; not executed here. |
| [Instruction backtranslation](https://arxiv.org/abs/2308.06259) | Learned inverse mapping, preserved source responses and iterative curation. Not executed here. |
| [Simula](https://arxiv.org/abs/2603.29791) | Deliberate dataset coverage and critique. This is Davidson and colleagues' published method, not an invention of this repository; not executed here. |
| [Attributed Grounding](https://arxiv.org/abs/2506.03968) | Grounding and diversification extension for the language program; not implemented or executed here. |
| [Evaluating Language Models as Synthetic Data Generators, AgoraBench](https://aclanthology.org/2025.acl-long.320/) | Evaluation design relevant to the original request. The official ACL title and Kim and colleagues' authorship were checked. Its language model benchmark was not run here. |
| [Active Synthetic Data Generation](https://arxiv.org/abs/2512.00884) | Student directed selection is relevant prior art. The static tabular portfolio is not a reproduction of this training loop. |
| [Absolute Zero](https://arxiv.org/abs/2505.03335) | Executable reasoning and self play are a separate research track. Not executed here. |
| [Persona Hub](https://arxiv.org/abs/2406.20094) | Diversity conditioning candidate for text. Not a local numerical baseline and not executed here. |
| [Dromedary](https://arxiv.org/abs/2305.03047) | Adjacent principle based alignment. Earlier extension recorded abstract screening rather than full reading in that pass. |
| [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) | Separate synthetic pretraining track. Earlier extension recorded abstract screening, not a reproduced training pipeline. |

## Evaluation and retained grounding

| Record and primary source | Role and execution boundary |
| :--- | :--- |
| [Best Practices and Lessons Learned on Synthetic Data](https://arxiv.org/abs/2404.07503) | Evaluation background. Ruibo Liu and colleagues' authorship was verified. This is not another generator to put on the local leaderboard. |
| [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y) | Relevant risk background. The earlier extension explicitly recorded unsuccessful full text retrieval in its independent pass; no complete reading is certified here. |
| [Is Model Collapse Inevitable?](https://arxiv.org/abs/2404.01413) | Accumulation versus replacement motivates separate augmentation conditions. The primary record identifies Matthias Gerstgrasser and colleagues, replacing an incomplete author placeholder in the earlier ledger. Not replicated here. |
| [Classification Accuracy Score for Conditional Generative Models](https://papers.nips.cc/paper_files/paper/2019/hash/fcf55a303b71b84d326fb1d06e332a26-Abstract.html) | Downstream utility motivation. Screening this concept does not validate the local benchmark as a universal metric. |
| [Risk In Context](https://arxiv.org/abs/2507.17066) | Privacy evaluation prior art. The current support distance screen is not a privacy assessment or a formal guarantee. |

## Executed tabular mechanisms

| Record and primary source | Local implementation and qualification |
| :--- | :--- |
| [Modeling Tabular Data using Conditional GAN](https://papers.nips.cc/paper_files/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Paper.pdf) | CTGAN and TVAE implemented and trained with local architectures and budgets. Method and architecture sections were inspected. Published scores were not reproduced. |
| [TabDDPM](https://proceedings.mlr.press/v202/kotelnikov23a.html) | Numerical class conditional diffusion implemented and trained. This is not the full mixed feature evaluation of the paper. |
| [SMOTE](https://doi.org/10.1613/jair.953) | Within class interpolation is the implemented simple baseline. The all class sampling policy is a local adaptation, not an exact reproduction of a minority oversampling experiment. |

## Stronger and adjacent tabular candidates

| Record and primary source | Decision |
| :--- | :--- |
| [TabSyn](https://arxiv.org/abs/2310.09656) | Relevant latent diffusion baseline. Earlier method section review is recorded, but there is no local implementation or result. |
| [TabDiff](https://arxiv.org/abs/2410.20626) | Relevant mixed modality diffusion baseline. Not implemented or executed; therefore the current study cannot claim a frontier comparison. |
| [Forest Diffusion](https://arxiv.org/abs/2309.09968) | Relevant CPU oriented tree based synthesis baseline. Not implemented or executed. |
| [Geometry Aware Tabular Diffusion](https://arxiv.org/abs/2606.02607) | Relevant recent relational supervision candidate. Primary metadata and abstract were checked again; not implemented or executed here. |
| [CoDi](https://proceedings.mlr.press/v202/lee23i.html) | Mixed feature diffusion track. Earlier screening only; not implemented or executed. |
| [GReaT](https://arxiv.org/abs/2210.06280) | Language models as tabular generators bridge the two programs. No pretrained weights were used in this extension. |
| [Generating Benchmark Health Data Using a Tabular Diffusion Transformer](https://arxiv.org/abs/2608.14496) | Primary abstract describes learning across heterogeneous tables, a different task from a generator trained on one numerical schema. Adjacent candidate, not a matched baseline. |
| [Task Conditioned Agricultural Data Generation](https://arxiv.org/abs/2607.09751) | Relevant task aware prior art combining a Bayesian network and a tabular foundation model. Primary abstract checked; agricultural task claims are not local findings. |
| [Empirical Marginal Copula Synthesis for Educational Data](https://arxiv.org/abs/2604.04195) | Adjacent application candidate. It was screened, not used as an implementation source or proof that the local copula preserves privacy. |

## Consequence for research claims

A cross paper leaderboard must separate compatible tasks and required resources. This study executes a small numerical classification track. It does not compare the six language model methods to tabular methods on an invented shared quality score. The four source portfolio is a local composition hypothesis, not established novelty. Its measured failure to beat simple controls must remain visible.

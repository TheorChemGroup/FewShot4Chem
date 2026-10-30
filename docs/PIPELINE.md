# Pipeline: data → features → models → evaluation → tables and figures

Every number in the paper's Figures 1 and 2 and in `tables/` is produced by the stages below.
`results/` holds the per-repeat outputs of the runs used for the paper, so each stage can be
re-run and compared with them (`scripts/check_reproduction.py`).

## 1. Data and cohorts — `scripts/make_cohorts.py`

Datasets are loaded with `utils.datasets.load_dataset` (MoleculeNet BACE, Lipophilicity, ESOL,
FreeSolv; QM9 with 129,425 molecules). All random draws use seed 2032025.

| Cohort | Used for | Size and split | File |
|---|---|---|---|
| small-sample | Figure 1; Figure 2 at 60 | 100 molecules per dataset (QM9: from the uniform 1,250); 15 repeats of 50 fit / 10 validation / 40 test | `data/cohorts/small_sample.json` |
| QM9 uniform 1,250 | Figure 2 at 1,000 | 900 / 100 / 250, 5 repeats | `runners/qm9_uniform_sample_1250.txt` |
| QM9 random 12,500 | Figure 2 at 10,000 | 9,000 / 1,000 / 2,500, 5 repeats | `data/cohorts/qm9_random12500_indices.txt` |
| QM9 random 125 (inside the 12,500) | Figure 2 at 100 | 90 / 10 / 25, 5 repeats | `data/cohorts/qm9_random125.json` |

`make_cohorts.py` documents each sampling rule and re-derives every shipped cohort and split.

## 2. Features — `scripts/featurize.py`, `scripts/qm9_curve.py featurize`

A frozen model (`models/configs/<family>/<model>.yaml`, wrappers in `nn/`) embeds each molecule
through `utils.embed_dataset`. RDKit descriptors are the 43 `rdMolDescriptors.Properties`
(`nn/descriptors.py`); Morgan fingerprints are 2,048-bit, radius 2 (`nn/fingerprint_creator.py`).
Gemini Embedding 2 vectors come from the embedding API (`nn/text_embedder.py`).

## 3. Models

**Frozen representation + head** (`validation/heads/`, hyperparameters in
`models/configs/heads/`): ridge / logistic regression with cross-validated regularisation, MLP,
XGBoost, sparse variational GPs with RBF and Matérn kernels, TabPFN and TabICL.
* Figure 1: `scripts/fit_head.py`. Features standardised on the 60 fit+validation molecules,
  head seed 2032025 + repeat; BACE uses class-balanced sample weights.
* Figure 2 at 100 and 10,000: `scripts/qm9_curve.py head`. Features standardised on the fit
  molecules, head seed = repeat. At 1,000: `runners/run_one_eval.py --protocol uniform`.

**Full fine-tuning and LoRA** (`runners/finetune_one.py`): the backbone plus a native head
(embedding → 256 → 256 → 1, GELU), AdamW (weight decay 0.01), MSE (BCE with logits for BACE),
the epoch with the best validation score is kept and scored once on the test set. LoRA ranks 4,
8 and 16 (α = 2r, dropout 0.05, lr 5e-4).
* Figure 1: `scripts/finetune_small_sample.py` (full-data regime: lr 5e-6, 20 epochs, patience
  6, batch 16; BACE and Lipophilicity batch 256). ChemFM-1B full fine-tuning ran in FP32 for
  repeats 5–14 of ESOL, FreeSolv and QM9 and in FP16 otherwise.
* Figure 2: `scripts/qm9_curve.py finetune` at 100 (low-data regime: lr 1e-5, 30 epochs,
  patience 8, batch 256) and 10,000 (full-data regime); at 1,000 `runners/finetune_one.py
  --split grid`.

**CheMeleon** (`runners/finetune_chemeleon.py`, Chemprop; separate `chemeleon` environment).

**General-purpose LLMs** (`scripts/query_llm.py`): one prompt per repeat lists the 60
labelled support molecules (SMILES and value) and asks for the 40 query molecules as strict
JSON. A shuffled-label control permutes the support labels. Prompts: `data/llm_table_probe/`.

## 4. Evaluation and selection

R² for regression, ROC-AUC for BACE (Figure 1 plots 2·AUC − 1). For each model and adaptation
the configuration (head, or LoRA rank) with the best **mean validation score** over all repeats
is selected and its test scores are reported:
* Figure 1: over the 15 repeats, among configurations whose 15 validation scores come from the
  saved, tested model (`eligible_for_validation_selection`; ESOL/FreeSolv/QM9 fine-tuning and LoRA
  runs recorded only the best training-epoch score and are not eligible); the winners are marked
  `selected_in_figure` in `results/figure1_repeats.csv` and re-derived by `scripts/select_figure1.py`;
* Figure 2: over the 5 repeats, among configurations scored on all five
  (`scripts/select_figure2.py`).
General-LLM responses are scored by `scripts/score_llm_responses.py`; refusals and invalid
answers are recorded as such. The GPT-6 astra point at support 1,000 in Figure 2 (1,000 labelled
molecules of the uniform 1,250 sample in the prompt, 250 queried) is re-scored by
`scripts/select_figure2.py` from `results/llm_uniform1250/`.

## 5. Tables and figures

| Output | Built by | From |
|---|---|---|
| `tables/Figure1_reproduction.xlsx` | `scripts/build_tables.py` | `results/figure1_repeats.csv` |
| `tables/Figure1_general_FMs.xlsx` | `scripts/build_tables.py` | `results/figure1_general_fm_repeats.csv` |
| `tables/Figure2_reproduction.xlsx` | `scripts/build_tables.py` | `results/figure2_points.csv`, `results/figure2_repeat_scores.csv` |
| `figures/figure{1,2,S2}.{png,pdf}` | `scripts/render_figures.py` | `figures/*.plotly.json` (checked against the tables by `scripts/check_figures.py`) |

## Limitations of the record

* Figure 2 at 1,000: fine-tuning/LoRA validation scores are the best per-epoch scores during
  training, not re-evaluations of the saved final model.
* Figure 2 at 10,000: within some LoRA configurations, repeats were trained with batch size 16
  and others with 256 (`batch_size` in `results/figure2_candidate_repeats.csv`).
* QM9 molecule records depend slightly on how the QM9 download is processed. The released
  pipeline reproduces the small-sample and 125-molecule RDKit/Morgan features and head scores
  exactly; at 10,000 the RDKit features of a few strained polycyclic molecules differ from the
  cache used for the paper, which moves the RDKit linear/XGBoost test R² by 0.0003-0.006.
* ChemFM-1B full fine-tuning in FP16 produced non-finite parameters after the first optimizer
  step; those repeats are reported as failures (blank scores), not as zeros.

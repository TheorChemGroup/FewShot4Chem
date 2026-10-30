# Are chemical foundation models really foundational?

Code, data splits, per-repeat results and reproduction tables for the paper. We compare frozen
and adapted chemical foundation models (ChemBERTa-2/3, MoLFormer, MoLFormer-XL, MoLFormer-c3,
ChemFM-1B, SELFormer, GIN, GROVER, CLAMP, CheMeleon), classic RDKit descriptors and Morgan
fingerprints, a general text-embedding model (Gemini Embedding 2) and table-prompted
general-purpose LLMs on BACE, Lipophilicity, ESOL, FreeSolv and QM9 (ε_LUMO, U₀).

## Contents

| Path | What |
|---|---|
| `tables/` | **Reproduction tables** (`.xlsx`): every repeat score behind Figures 1 and 2 |
| `results/` | The same per-repeat results as CSV, all Figure 2 candidate configurations, the raw LLM responses (`llm_responses/`, provider-encrypted reasoning removed) and the GPT-6 astra predictions at Figure 2 support 1,000 (`llm_uniform1250/`) |
| `figures/` | Figures 1, 2 and S2 as Plotly specifications (data + styling) |
| `data/cohorts/` | The fixed molecule cohorts and splits; `data/llm_table_probe/` the LLM prompts |
| `scripts/` | One entry point per stage (below) |
| `nn/`, `utils/`, `validation/`, `vis/` | Model wrappers, dataset loaders, prediction heads, plotting helpers |
| `runners/finetune_one.py`, `finetune_chemeleon.py`, `run_one_eval.py`, `uniform_protocol.py` | Fine-tuning / LoRA runner, CheMeleon fine-tuning, uniform-protocol head evaluation |
| `models/configs/` | Model and head hyperparameters |
| `docs/PIPELINE.md` | How each number was produced: data → features → models → evaluation → tables |

## Installation

GPU (the environment used for the paper; CUDA 11.8, Linux):

```sh
pixi install                                  # environment "default"; see pyproject.toml
pixi run install-fast-transformers            # only for the original MoLFormer (config `molformer`)
pixi shell
```

CPU only (enough for everything under "Reproducing the tables and figures", featurising with
RDKit/Morgan, and the classical heads). With [uv](https://docs.astral.sh/uv/), which also provides
Python 3.10 if it is not installed:

```sh
uv venv --python 3.10 .venv && . .venv/bin/activate
uv pip install -r requirements-cpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

or with pip and an existing Python 3.10:

```sh
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Use uv or pip for the CPU environment: without a lock file pixi resolves every environment in
`pyproject.toml`, including the CUDA one, so it downloads the CUDA toolkit even for `-e cpu`.
Rendering figures needs Chrome or Chromium for kaleido; run `plotly_get_chrome` once if neither
is installed.

Two models need packages that build from Git, installed after the environment:

```sh
# CLAMP (GPL-3.0); -c keeps the versions above
pip install "clamp @ git+https://github.com/OstarkovSN/clamp.git" -c requirements-cpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
# the original MoLFormer: compiles against the installed torch (needs a C++ compiler)
pip install --no-build-isolation \
    "pytorch-fast-transformers @ git+https://github.com/idiap/fast-transformers@2ad36b97e64cb93862937bd21fcc9568d989561f"
```

(with uv, prefix both with `uv`). In the pixi environments CLAMP is already included.

Model weights download on first use from their public sources (Hugging Face, Zenodo). Two models
need files placed by hand:

* GIN: `supervised_contextpred.pth` from <https://github.com/snap-stanford/pretrain-gnns> into
  `models/weights/gin/` (SHA-256 `107197f159e9a26ed9a026b655560ac341d2d7ddec4a8d2540e83c9f87e256ac`).
* The original MoLFormer (config `molformer`; MoLFormer-XL and -c3 load from Hugging Face):
  `nn/molformer/wrapper.py` downloads IBM's released checkpoint and encoder file; the tokenizer
  vocabulary comes from IBM's Hugging Face tokenizer:

  ```sh
  python scripts/get_molformer_vocab.py     # writes models/weights/molformer/vocab.json
  ```

  Files used for the paper (SHA-256): `N-Step-Checkpoint_3_30000.ckpt`
  `60f0acac8e3cf7c6b521ab52e1e09756b81b48c977b97426be0f695961a38be5`, `vocab.json`
  `61e31e6eb0448d2e7bb442d562f20df9be818a20294d0a70dd395783e095abcd` (the script checks the
  content, which is independent of key order), `encoder/pubchem_canon_zinc_final_vocab_sorted.pth`
  `8fe7334da673dc3b4cb1df34b6ca29638eae00a3cfbd71d638a86a379523fd38`.

CheMeleon needs the separate `chemeleon` pixi environment (chemprop requires Python ≥ 3.11).
TabPFN 9.2.0 (v3.5 weights) and TabICL 2.2.0 (v2 weights) were used with local checkpoints; pass
them to `scripts/fit_head.py` with `--tabpfn-weights` / `--tabicl-weights`.

Datasets (MoleculeNet BACE, Lipophilicity, ESOL, FreeSolv; QM9) download automatically into
`datasets/` the first time a script loads them.

## Reproducing the tables and figures

From the shipped per-repeat results (seconds, in the CPU environment):

```sh
python scripts/build_tables.py           # tables/*.xlsx from results/*.csv
python scripts/select_figure1.py --check  # every Figure 1 box from the selection rule
python scripts/select_figure2.py --check results/figure2_points.csv   # Figure 2 selection rule
python scripts/check_figures.py          # every plotted value equals the tables
python scripts/render_figures.py         # figures/*.png, *.pdf (plotly 6.9.0 + kaleido 1.3.0; `plotly_get_chrome` once)
python scripts/make_cohorts.py           # every cohort/split regenerated from its sampling rule
```

From raw data (examples; each prints validation and test scores per repeat):

```sh
# Figure 1, frozen representation + head (CPU for RDKit / Morgan)
python scripts/featurize.py --dataset ESOL --model-config models/configs/descriptors/rdkit.yaml
python scripts/fit_head.py --dataset ESOL --model rdkit-descriptors --head linear --out results/mine.jsonl
python scripts/check_reproduction.py heads results/mine.jsonl     # compare with the paper's rows

# Figure 1, fine-tuning / LoRA / CheMeleon (GPU)
python scripts/finetune_small_sample.py --dataset ESOL --model-config models/configs/chemberta/chemberta-2-77M-MTR.yaml --repeat 0
python scripts/finetune_small_sample.py --dataset ESOL --model-config models/configs/chemberta/chemberta-2-77M-MTR.yaml --lora-rank 8 --repeat 0
python scripts/finetune_small_sample.py --dataset ESOL --chemeleon --repeat 0

# Figure 1, general-purpose LLMs: re-score the stored responses, or query again (paid API)
python scripts/score_llm_responses.py && python scripts/check_reproduction.py llm results/llm_rescored.csv
OPENROUTER_API_KEY=... python scripts/query_llm.py --model openai/gpt-6-astra --repeats 0-14 --out-dir results/my_responses
python scripts/score_llm_responses.py --responses results/my_responses --out results/my_scores.csv

# Figure 2 (QM9 ε_LUMO learning curve) at support 100 and 10,000
python scripts/qm9_curve.py cohort
python scripts/qm9_curve.py featurize --model-config models/configs/descriptors/rdkit.yaml
python scripts/qm9_curve.py head --support 10000 --model rdkit-descriptors --head xgboost
python scripts/qm9_curve.py finetune --support 10000 --model-config models/configs/selformer/selformer.yaml --lora-rank 8 --repeat 0
# ... and at support 1,000 (uniform 1,250 sample); about 20 min on CPU, see below
python runners/run_one_eval.py --protocol uniform --dataset QM9 --seed 0 -vv \
    --model-config models/configs/descriptors/rdkit.yaml --head-config models/configs/heads/gp-rbf.yaml
python runners/finetune_one.py --split grid --regime full-data --lr-schedule none \
    --target QM9.eps_lumo --seed 0 --model-config models/configs/selformer/selformer.yaml
```

The `run_one_eval.py` example took 20 minutes on an 8-core CPU (2.4 GB peak memory): about
6 minutes to download and process QM9 and compute the descriptors, 4 minutes to fit the GP on all
16 QM9 targets, and 10 minutes to draw the per-target plots under `results_uniform/`. Without `-v`
it prints only the QM9 processing bar; `-vv` adds a log line at each stage (and the plotting
library's own messages). Its ε_LUMO test R² should be close to the paper's repeat 0 (0.697,
`results/figure2_candidate_repeats.csv`); our CPU run gave 0.698 (the GP fit is not bit-identical
across hardware).

`scripts/featurize.py --model-config` takes any config in `models/configs/` (chemical FMs need a
GPU for reasonable speed; Gemini embeddings need a paid API).

## Protocol in brief

* **Figure 1** (small-sample): per dataset a fixed cohort of 100 molecules; 15 repeats, each a
  different 50 fit / 10 validation / 40 test split. For every model and adaptation (frozen +
  head, full fine-tuning, LoRA rank 4/8/16) the box shows the configuration with the best mean
  validation score over the 15 repeats; test scores are plotted. Only configurations whose
  validation scores come from the saved, tested model are eligible; ESOL/FreeSolv/QM9
  fine-tuning and LoRA runs recorded only their best training-epoch validation score and are
  therefore not eligible (column `eligible_for_validation_selection`). BACE: ROC-AUC (plotted as
  2·AUC − 1); otherwise R².
* **Figure 2** (QM9 ε_LUMO): 5 repeats at 100, 1,000 and 10,000 training + validation molecules
  (plus the Figure 1 cohort at 60); each point is the mean test R² of the configuration with the
  best mean validation R².
* General-purpose LLMs see 60 labelled molecules in the prompt and predict 40; a shuffled-label
  control is run alongside. Refusals and invalid outputs are reported, never scored as zero.

Known limitations of the record, also listed in `docs/PIPELINE.md`: at support 1,000 the
fine-tuning/LoRA validation scores are the best training-epoch scores (not re-evaluated from the
saved final state); within some support-10,000 LoRA configurations repeats were trained with
batch size 16 and others with 256 (column `batch_size` of `results/figure2_candidate_repeats.csv`);
at support 10,000 the RDKit features of a few QM9 molecules differ between QM9 processings
(test R² within 0.006 of the published values; the smaller cohorts reproduce exactly);
ChemFM-1B full fine-tuning diverged (non-finite parameters after the first optimizer step) in
FP16 and those repeats are reported as failures.

## Tests

```sh
pytest -m "not slow"
```

## Licence

MIT (see `LICENSE`). Code taken from GROVER, MoLFormer, pretrain-gnns and CheMeleon keeps its
original licence, listed in `LICENSE`. CLAMP, installed as a dependency, is GPL-3.0.

"""Full fine-tuning, LoRA and CheMeleon fine-tuning on the small-sample (Figure 1) cohorts.

The training code is `runners/finetune_one.py` (chemical FMs: native head
emb→256→256→1, AdamW, best-validation checkpoint) and `runners/finetune_chemeleon.py`
(CheMeleon / Chemprop). Those runners load a dataset and draw a split themselves; for Figure 1
they were run on the fixed 100-molecule cohorts instead. This wrapper does exactly that:

* the dataset each runner loads is the cohort's 100 records (SMILES and labels), in cohort order;
* the runner's split is the cohort split for the repeat (50 fit / 10 validation / 40 test);
* the runner's standard options for this protocol: `--regime full-data --split grid
  --lr-schedule none` (+ `--use-lora --lora-rank r`), and CheMeleon `--protocol uniform`.

ESOL, FreeSolv and QM9 (ε_LUMO, U₀) ran with batch size 16. The BACE and Lipophilicity full
fine-tunes ran later with batch size 256 (one full-batch step per epoch on 50 molecules) and stop
on a non-finite loss or parameters; this wrapper applies that recipe for those two datasets.

    python scripts/finetune_small_sample.py --dataset ESOL --model-config \
        models/configs/chemberta/chemberta-2-77M-MTR.yaml --repeat 0
    python scripts/finetune_small_sample.py --dataset ESOL --model-config ... --lora-rank 8 --repeat 0
    python scripts/finetune_small_sample.py --dataset ESOL --chemeleon --repeat 0
"""

import json
import os
import sys
import types
from pathlib import Path

import click
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runners")]

COHORTS = ROOT / "data/cohorts/small_sample.json"
QM9_COLUMNS = [
    "mu", "alpha", "epsilon_homo", "epsilon_lumo", "gap_homo_lumo", "R2_avg", "ZPVE", "U_0",
    "U_298", "H_298", "G_298", "C_v_298", "U_0_atom", "U_298_atom", "H_298_atom", "G_298_atom",
]
LABEL = {"BACE": "active", "Lipophilicity": "logD", "ESOL": "logS", "FreeSolv": "dG_hydr"}
# finetune_one.py target names
TARGET = {"ESOL": "ESOL", "FreeSolv": "FreeSolv", "BACE": "BACE", "Lipophilicity": "Lipophilicity",
          ("QM9", "epsilon_lumo"): "QM9.eps_lumo", ("QM9", "U_0"): "QM9.U_0"}
BATCH_256 = {"BACE", "Lipophilicity"}


def use_cohorts_as_datasets(cohorts):
    """Make `utils.datasets.load_dataset(name)` return the cohort's 100 records."""

    def load_dataset(name, *_, **__):
        cols = QM9_COLUMNS if name == "QM9" else [LABEL[name]]
        return [
            types.SimpleNamespace(smiles=r["smiles"], y=np.array([float(r[c]) for c in cols], dtype=np.float32))
            for r in cohorts[name]["records"]
        ]

    module = types.ModuleType("utils.datasets")
    module.load_dataset = load_dataset
    module.maybe_load_targets = lambda ds: np.stack([d.y for d in ds])
    module.TARGETS_NAMES = {name: (QM9_COLUMNS if name == "QM9" else [name]) for name in cohorts}
    sys.modules["utils.datasets"] = module


def use_cohort_splits(cohorts):
    """Route both runners' split functions to the stored cohort split of each repeat."""
    import uniform_protocol

    # Splits are drawn from the repeat number only, so they are identical for every dataset.
    splits = cohorts["QM9"]["splits"]

    def split(n, repeat):
        assert n == 100, n
        s = splits[str(repeat)]
        return tuple(np.asarray(s[k]) for k in ("fit", "val", "test"))

    uniform_protocol.QM9_N_TOTAL = 100
    uniform_protocol.qm9_sample = lambda: np.arange(100)
    uniform_protocol.grid_split = split
    return split


def legacy_molformer_compat():
    """Load the original MoLFormer checkpoint for fine-tuning without changing its weights.

    * the pretraining checkpoint's trainer RNG is not restored (it would overwrite the repeat seed);
    * tokenizer special-token ids are taken from the shipped vocabulary;
    * on GPU the token tensors follow the model's device (the CPU forward is unchanged).
    """
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch
    from fast_transformers.masking import LengthMask

    from nn.molformer.original import LightningModule as OriginalMolformer
    from nn.molformer.tokenizer import MolTranBertTokenizer
    from nn.molformer.wrapper import LightningMoLFormer

    OriginalMolformer.on_load_checkpoint = lambda self, checkpoint: None

    original_init = MolTranBertTokenizer.__init__

    def init_with_vocab_specials(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        old = dict(self._added_tokens_decoder)
        self._added_tokens_encoder = {str(t): self.vocab[str(t)] for t in old.values()}
        self._added_tokens_decoder = {self.vocab[str(t)]: t for t in old.values()}

    MolTranBertTokenizer.__init__ = init_with_vocab_specials

    original_forward = LightningMoLFormer.forward

    def forward_on_model_device(self, data):
        device = next(self.model.parameters()).device
        if device.type == "cpu":
            return original_forward(self, data)
        batch = self.tokenizer.batch_encode_plus(data.smiles, padding=True, add_special_tokens=True)
        idx = torch.tensor(batch["input_ids"], device=device)
        mask = torch.tensor(batch["attention_mask"], device=device)
        tokens = self.model.blocks(self.model.tok_emb(idx), length_mask=LengthMask(mask.sum(-1)))
        expanded = mask.unsqueeze(-1).expand(tokens.size()).float()
        return torch.sum(tokens * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)

    LightningMoLFormer.forward = forward_on_model_device


def chemfm_in_fp32():
    """ChemFM-1B full fine-tuning in FP32 (repeats 5-14; repeats 0-4 ran in FP16 and diverged)."""
    from nn.chemfm import LightningChemFM

    LightningChemFM.prepare_for_finetuning = lambda self: self.float()


@click.command()
@click.option("--dataset", type=click.Choice(["BACE", "Lipophilicity", "ESOL", "FreeSolv", "QM9"]), required=True)
@click.option("--property", "prop", type=click.Choice(["epsilon_lumo", "U_0"]), default=None,
              help="QM9 only.")
@click.option("--model-config", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--lora-rank", type=int, default=None, help="LoRA instead of full fine-tuning.")
@click.option("--chemeleon", is_flag=True, help="CheMeleon (Chemprop) fine-tuning.")
@click.option("--repeat", type=int, required=True)
@click.option("--output-dir", default=str(ROOT / "results/small_sample_adaptations"), show_default=True)
def main(dataset, prop, model_config, lora_rank, chemeleon, repeat, output_dir):
    if dataset == "QM9" and prop is None:
        raise click.UsageError("--property is required for QM9")
    if chemeleon and dataset in BATCH_256:
        raise click.UsageError("CheMeleon was fine-tuned on ESOL, FreeSolv and QM9 only")
    if chemeleon == bool(model_config):
        raise click.UsageError("give exactly one of --model-config and --chemeleon")
    cohorts = json.loads(COHORTS.read_text())
    use_cohorts_as_datasets(cohorts)
    split = use_cohort_splits(cohorts)

    if chemeleon:
        import finetune_chemeleon

        # finetune_chemeleon seeds its split with 2032025 + repeat.
        finetune_chemeleon._build_split = lambda n, seed: split(n, seed - 2032025)  # noqa: SLF001
        args = ["--dataset", dataset, "--seed", str(repeat), "--protocol", "uniform", "--out-dir", output_dir]
        if dataset == "QM9":
            args += ["--target", prop]
        finetune_chemeleon.main.main(args=args, standalone_mode=False)
        return

    # Installed for every chemical-FM run, as in the paper runs; it only touches MoLFormer classes.
    legacy_molformer_compat()
    # As run: ChemFM full fine-tuning of ESOL/FreeSolv/QM9 repeats 5-14 in FP32; FP16 otherwise.
    if (lora_rank is None and dataset not in BATCH_256 and repeat >= 5
            and OmegaConf.load(model_config).name == "chemfm-1b"):
        chemfm_in_fp32()
    import finetune_one

    target = TARGET[(dataset, prop)] if dataset == "QM9" else TARGET[dataset]
    args = ["--model-config", model_config, "--target", target, "--seed", str(repeat),
            "--regime", "full-data", "--split", "grid", "--lr-schedule", "none", "--output-dir", output_dir]
    if lora_rank:
        args += ["--use-lora", "--lora-rank", str(lora_rank)]
    if dataset in BATCH_256:
        if lora_rank:
            raise click.UsageError("BACE/Lipophilicity were run with full fine-tuning only")
        args += ["--batch-size", "256", "--stop-on-nonfinite"]
    finetune_one.main.main(args=args, standalone_mode=False)


if __name__ == "__main__":
    main()

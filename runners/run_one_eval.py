"""Worker: run one (model, head, dataset, seed) evaluation cell for the main grid.

Usage:
  pixi run python runners/run_one_eval.py \\
    --model-config models/configs/chemberta/chemberta-2-77M-MTR.yaml \\
    --head-config models/configs/heads/gp-rbf.yaml \\
    --dataset FreeSolv \\
    --seed 0
"""

import gc
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model-config", "model_config_path", required=True,
              type=click.Path(exists=True, resolve_path=True))
@click.option("--head-config", "head_config_path", required=True,
              type=click.Path(exists=True, resolve_path=True))
@click.option("--dataset", required=True)
@click.option("--seed", default=0, type=int)
@click.option("--recalculate", is_flag=True, default=False)
@click.option("--max-elements", default=-1, type=int)
@click.option("--split-strategy", default="random", type=click.Choice(["random", "scaffold"]))
@click.option("--protocol", type=click.Choice(["grid", "uniform"]), default="grid",
              help="uniform = the uniform protocol: QM9 on the seeded 1,250 sample, validation scores recorded, "
                   "results under results_uniform/. The default reproduces the published grid.")
@click.option("--verbose", "-v", count=True)
def run_one_eval(
    model_config_path: str,
    head_config_path: str,
    dataset: str,
    seed: int,
    recalculate: bool,
    max_elements: int,
    split_strategy: str,
    protocol: str,
    verbose: int,
) -> None:
    """Run one evaluation cell: one model × one head × one dataset × one seed."""
    logging.basicConfig(
        level=max(10, 40 - verbose * 10),
        format="[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
    )

    try:
        from omegaconf import OmegaConf

        import validation.heads.gp_matern  # noqa: F401
        import validation.heads.gp_rbf  # noqa: F401
        import validation.heads.gp_tanimoto  # noqa: F401
        import validation.heads.linear  # noqa: F401
        import validation.heads.mlp  # noqa: F401
        import validation.heads.tabicl  # noqa: F401
        import validation.heads.tabpfn  # noqa: F401
        import validation.heads.xgboost  # noqa: F401

        RANDOM_SEED = 2032025
        model_cfg = OmegaConf.load(model_config_path)
        head_cfg = OmegaConf.load(head_config_path)
        if "seed" in head_cfg:
            head_cfg.seed = RANDOM_SEED + seed
        head_cfg.split_strategy = split_strategy
    # Propagate the cap so capped runs use a separate result directory.
        if max_elements != -1:
            head_cfg.max_elements = max_elements
        head_cfg.protocol = protocol

    # report.txt is written after metrics, marking a completed cell. Split
    # strategies use separate result directories.
        head_type = head_cfg.get("head_type", "unknown-head")
        base = Path("results") / dataset / model_cfg.name / str(head_type)
        if split_strategy != "random":
            base = base / f"split-{split_strategy}"
        if protocol == "uniform":
    # Use the validator's result path, including any training-size cap.
            base = Path("results_uniform") / dataset / model_cfg.name / str(head_type)
            if max_elements != -1:
                base = base / f"n_train-{max_elements}"
        signature = base / "metrics" / str(seed) / "report.txt"
        if signature.exists() and not recalculate:
            logger.warning("Skip: result already exists at %s", signature)
            return

        from validation.select_validator import select_validator

        # Use the CLI --seed as the saved run_id offset so each per-seed task writes to
        # its own metrics/<seed>/ dir; the validator's internal loop then sees
        # run_id == seed (with n_runs=1), keeping split/seed-threading semantics identical
        # to running n_runs=5 in a single process.
        validator = select_validator(head_cfg, dataset, random_seed=RANDOM_SEED)
        validator.n_runs = 1
        validator.start_run_id = seed
        validator.validate(model_cfg, recalculate=recalculate, max_elements=max_elements)

        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            logger.warning("torch not available, skipping GPU cleanup")

        gc.collect()
        logger.info(
            "Done: model=%s head=%s dataset=%s seed=%s",
            model_cfg.name, head_cfg.head_type, dataset, seed,
        )
    except Exception:
        logger.exception("run_one_eval failed (model=%s head=%s dataset=%s seed=%s)",
                         model_config_path, head_config_path, dataset, seed)
        raise SystemExit(1)


if __name__ == "__main__":
    run_one_eval()

"""Provides abstract target validator class."""

import json
import logging
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from umap import UMAP

from utils.datasets import TARGETS_NAMES, get_filtered_targets_names, maybe_load_targets
from utils.embed_dataset import EmbeddingParams, embed_dataset
from utils.filter_kwargs import filter_kwargs
from utils.load_models import load_model
from utils.minor import safe_write_image
from validation.abstract import Validator
from validation.heads import Head
from validation.splits import get_split_indices
from validation.utils import recursively_listify
from vis import plot_pred_vs_true_dimension_reduction_results
from vis.defaults import DEFAULT_STYLE

logger = logging.getLogger(__name__)

PROVIDE_DATASET_NAME = "Provide a dataset name explicitly or set dataset.external_name"


class TargetValidator(Validator):
    """Abstract TargetValidator delegating model fit/predict to a `Head`."""

    metric_functions: ClassVar[dict[str, Callable]] = {}
    formatters: ClassVar[dict[str, Callable[[float | np.floating], str]]] = {}
    metrics_to_display: ClassVar[set[str]] = set()

    def __init__(
        self,
        head_config: OmegaConf,
        dataset: Sequence[Data],
        random_seed: int = 42,
        dataset_name: str | None = None,
        n_runs: int = 5,
        confidence: float = 0.95,
    ) -> None:
        self.random_seed = random_seed
        self.head_config = head_config
        self.dataset = dataset
        try:
            if dataset_name is None:
                dataset_name = self.dataset.external_name
        except AttributeError as exc:
            raise ValueError(PROVIDE_DATASET_NAME) from exc
        self.dataset_name = dataset_name
        self.n_runs = n_runs
        self.confidence = confidence
        self.targets_names = TARGETS_NAMES[self.dataset_name]
        # Offset for run_id labels in saved output paths. Used by the per-seed grid
        # to write each task to its own metrics/<seed>/ dir instead of clobbering metrics/0/.
        self.start_run_id = 0

    def _result_root(self, model_config: OmegaConf) -> Path:
        """Per-cell directory. Includes head_type so different heads (gp-matern,
        xgboost, mlp, ...) on the same (dataset, model) don't clobber each other.
        Layout: results/<dataset>/<model>/<head_type>/{metrics,plots}/...
        Non-random split strategies append .../split-<strategy>/ and non-default
        max_elements appends .../n_train-<N>/ so the scaffold sub-grid and the
        low-data sweep don't clobber or silently dedupe against the full-data
        random-split main grid.
        """
        head_type = self.head_config.get("head_type", "unknown-head")
        # Keep uniform protocol results in a separate tree.
        root = "results_uniform" if self.head_config.get("protocol") == "uniform" else "results"
        base = Path(root) / self.dataset_name / model_config.name / str(head_type)
        strategy = self.head_config.get("split_strategy", "random")
        if strategy != "random":
            base = base / f"split-{strategy}"
        max_elements = self.head_config.get("max_elements", -1)
        if max_elements is not None and max_elements != -1:
            base = base / f"n_train-{max_elements}"
        return base

    def load_embeddings_and_targets(
        self,
        model_config: OmegaConf,
        *,
        max_elements: int = -1,
        recalculate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings_path = Path("datasets") / self.dataset_name / f"embeddings_{model_config.name}.npy"
        # Cached embeddings do not require model construction or its dependencies.
        # Cache writes are atomic, so readers do not observe partial files.
        if not recalculate and embeddings_path.exists():
            logger.info("Embeddings found at %s, skipping model load", embeddings_path)
            embeddings = np.load(embeddings_path)
        else:
            model = load_model(model_config)
            logger.debug("Embedding dataset %s", self.dataset_name)
            embeddings = embed_dataset(
                model,
                self.dataset,
                embedding_params=EmbeddingParams(
                    path=embeddings_path,
                    batch_size=model_config.batch_size or 1,
                    recalculate=recalculate,
                ),
            )
        targets = maybe_load_targets(self.dataset)
        if self.head_config.get("protocol") == "uniform" and self.dataset_name == "QM9":
            # Sample QM9 across the full ordered dataset to avoid prefix bias.
            # The max-elements value still names the result directory.
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runners"))
            import uniform_protocol as _UP
            if len(embeddings) != _UP.QM9_N_TOTAL or len(targets) != _UP.QM9_N_TOTAL:
                raise SystemExit("QM9: %d embeddings / %d targets, the sample is drawn from %d"
                                 % (len(embeddings), len(targets), _UP.QM9_N_TOTAL))
            _idx = _UP.qm9_sample()
            embeddings, targets = embeddings[_idx], targets[_idx]
        else:
            if max_elements == -1:
                max_elements = len(embeddings)
            embeddings = embeddings[:max_elements]
            targets = targets[:max_elements]
        if targets.ndim == 3:  # noqa: PLR2004
            targets = targets.reshape(targets.shape[0], -1)
        return embeddings, targets

    def _get_smiles_list(self, n: int) -> list[str]:
        """Extract `.smiles` from each PyG Data object in self.dataset, truncated to n."""
        smiles: list[str] = []
        for i, data in enumerate(self.dataset):
            if i >= n:
                break
            s = getattr(data, "smiles", "")
            smiles.append(s if isinstance(s, str) else "")
        if len(smiles) != n:
            msg = f"_get_smiles_list: requested n={n} but dataset yielded {len(smiles)} entries"
            raise ValueError(msg)
        return smiles

    def prepare_data(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        random_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if random_seed is None:
            random_seed = self.random_seed
        strategy = self.head_config.get("split_strategy", "random")
        smiles_list = self._get_smiles_list(len(embeddings)) if strategy != "random" else None
        train_idx, test_idx = get_split_indices(
            len(embeddings),
            strategy=strategy,
            test_size=0.2,
            seed=random_seed,
            smiles_list=smiles_list,
        )
        x_train, x_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        # gp-tanimoto requires binary 0/1 input. StandardScaler would z-score that into
        # floats and the head's fit() rejects non-binary input with NotImplementedError,
        # which the validator turns into "All N runs were skipped". Pass through unscaled
        # so the binary invariant survives.
        if self.head_config.get("head_type") == "gp-tanimoto":
            return x_train, x_test, y_train, y_test
        scaler = StandardScaler()
        scaler.fit(x_train)
        return scaler.transform(x_train), scaler.transform(x_test), y_train, y_test

    @staticmethod
    def _maybe_subset_targets(
        targets: np.ndarray,
        target_indices: list[int] | None,
    ) -> np.ndarray:
        """Subset targets along axis=1 if target_indices is provided; pass-through otherwise."""
        if target_indices is None:
            return targets
        return targets[:, list(target_indices)]

    @abstractmethod
    def create_head(self) -> Head:
        """Create a Head instance from self.head_config."""
        raise NotImplementedError

    def get_fit_kwargs(
        self,
        x_train: np.ndarray,  # noqa: ARG002
        x_val: np.ndarray,  # noqa: ARG002
        y_train: np.ndarray,  # noqa: ARG002
        y_val: np.ndarray,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Subclasses override to add per-task fit kwargs (e.g. sample_weight).

        Called AFTER the validator carves the 90/10 val split, so y_train/y_val
        are the carved partitions — use them to compute correctly-sized weights.
        """
        return {}

    def get_additional_predictions(
        self,
        head: Head,
        x_test: np.ndarray,
        *,
        is_binary: bool = False,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        proba = head.predict_proba(x_test)
        if proba is None:
            return {}
        if is_binary:
            return {"y_score": proba[:, 1].reshape(-1, 1, 1)}
        return {"y_score": proba}

    def calculate_metrics(
        self,
        y_test: np.ndarray,
        predicted: np.ndarray,
        **additional_predictions: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        metric_kwargs = {"y_true": y_test, "y_pred": predicted, **additional_predictions}
        metrics: dict[str, Any] = {}
        for metric_name, metric_function in self.metric_functions.items():
            metrics[metric_name] = metric_function(**filter_kwargs(metric_function, metric_kwargs))
        return metrics

    def _save_validation_metrics(self, model_config, head, x_val, y_val, run_id, is_binary, kwargs):
        pred = head.predict(x_val)
        extra = self.get_additional_predictions(head, x_val, **filter_kwargs(self.get_additional_predictions, kwargs))
        yv = y_val
        if is_binary:
            pred = pred.reshape(-1, 1) if pred.ndim == 1 else pred
            yv = yv.reshape(-1, 1) if getattr(yv, "ndim", 1) == 1 else yv
        metrics = self.calculate_metrics(yv, pred, **extra, **filter_kwargs(self.calculate_metrics, kwargs))
        folder = self._result_root(model_config) / "metrics" / str(run_id)
        folder.mkdir(parents=True, exist_ok=True)
        # Same rule as save_metrics_for_run: arrays to .npy, lists (roc_curve, precision_recall_curve --
        # ragged per target) to .json. np.asarray on those raised, failing every classification cell.
        for name, value in metrics.items():
            if isinstance(value, list):
                with (folder / ("val_%s.json" % name)).open("w", encoding="utf-8") as f:
                    json.dump(recursively_listify(value, convert_numbers_to_python_types=True), f, ensure_ascii=False)
            else:
                np.save(folder / ("val_%s.npy" % name), np.asarray(value))
        np.save(folder / "val_n.npy", np.asarray(len(yv)))

    def save_metrics_for_run(
        self,
        model_config: OmegaConf,
        metrics: dict[str, Any],
        run_id: int,
    ) -> None:
        metrics_folder = self._result_root(model_config) / "metrics" / str(run_id)
        metrics_folder.mkdir(parents=True, exist_ok=True)
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Target", style="dim", justify="center")
        for metric_name in self.metric_functions:
            if metric_name in self.metrics_to_display:
                table.add_column(metric_name, justify="center")
        for i, target in enumerate(self.targets_names):
            table.add_row(
                target,
                *[
                    self.formatters.get(name, lambda x: f"{x:.3f}")(values[i])
                    for name, values in metrics.items()
                    if name in self.metrics_to_display
                ],
            )
        if logger.isEnabledFor(logging.INFO):
            Console().print(table)
        with (metrics_folder / "report.txt").open("w", encoding="utf-8") as f:
            Console(file=f).print(table)
        for name, value in metrics.items():
            if isinstance(value, np.ndarray):
                np.save(metrics_folder / f"{name}.npy", value)
            elif isinstance(value, list):
                with (metrics_folder / f"{name}.json").open("w", encoding="utf-8") as f:
                    json.dump(recursively_listify(value, convert_numbers_to_python_types=True), f, ensure_ascii=False)

    def get_global_metrics_and_errors(
        self,
        metrics_per_run: list[dict[str, Any]],
    ) -> tuple[dict, dict]:
        metrics: dict[str, list] = defaultdict(list)
        for run in metrics_per_run:
            for name, value in run.items():
                metrics[name].append(value)
        keep = {}
        for name, values in metrics.items():
            try:
                keep[name] = np.stack(values, axis=0)
            except ValueError:
                logger.warning(
                    "Metric %r skipped from global aggregation: shapes inconsistent across runs %s",
                    name,
                    [np.asarray(v).shape for v in values],
                )
        means = {name: np.mean(arr, axis=0) for name, arr in keep.items()}
        scale = np.sqrt(self.n_runs) * norm.ppf(self.confidence)
        errors = {name: np.std(arr, axis=0) / scale for name, arr in keep.items()}
        return means, errors

    def save_metrics_global(
        self,
        model_config: OmegaConf,
        global_metrics: dict[str, Any],
        errors: dict[str, Any],
    ) -> None:
        folder = self._result_root(model_config) / "metrics" / "global"
        folder.mkdir(parents=True, exist_ok=True)
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Target", style="dim", justify="center")
        for name in self.metric_functions:
            if name in self.metrics_to_display:
                table.add_column(name, justify="center")
        for i, target in enumerate(self.targets_names):
            table.add_row(
                target,
                *[
                    self.formatters.get(n, lambda x: f"{x:.3f}")(v[i])
                    + " ±"
                    + self.formatters.get(n, lambda x: f"{x:.3f}")(errors[n][i])
                    for n, v in global_metrics.items()
                    if n in self.metrics_to_display
                ],
            )
        with (folder / "report.txt").open("w", encoding="utf-8") as f:
            Console(file=f).print(table)
        for name, metric in global_metrics.items():
            if isinstance(metric, np.ndarray):
                np.save(folder / f"{name}.npy", metric)
                np.save(folder / f"{name}_error.npy", errors[name])

    def save_data_for_run(
        self,
        model_config: OmegaConf,
        targets: np.ndarray,
        predicted: np.ndarray,
        run_id: int,
    ) -> None:
        folder = self._result_root(model_config) / str(run_id)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "targets.npy", targets)
        np.save(folder / "predictions.npy", predicted)

    def save_head_for_run(
        self,
        model_config: OmegaConf,
        head: Head,
        run_id: int,
    ) -> None:
        path_template = self.head_config.get("path", None)
        if path_template is None:
            logger.debug("head_config.path not set; skipping head save")
            return
        save_path = Path(
            path_template.format(
                model_name=model_config.name,
                dataset_name=f"{self.dataset_name}-run_{run_id}",
            ),
        )
        head.save(save_path)
        logger.info("Head saved to %s", save_path)

    @abstractmethod
    def plot_results_for_run(
        self,
        model_config: OmegaConf,
        targets: np.ndarray,
        predictions: np.ndarray,
        metrics: dict[str, Any],
        run_id: int,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    def plot_results_global(
        self,
        model_config: OmegaConf,
        global_metrics: dict[str, Any],  # noqa: ARG002
        embeddings: np.ndarray,
        style: dict[str, Any] = DEFAULT_STYLE,
        model_to_compare: str | None = None,
    ) -> None:
        plots_path = self._result_root(model_config) / "plots" / "global"
        if not model_to_compare:
            return
        try:
            comp_path = Path("datasets") / self.dataset_name / f"embeddings_{model_to_compare}.npy"
            comp_emb = np.load(comp_path)
            for name, model in (("umap", UMAP(n_components=2)), ("pca", PCA(n_components=2))):
                fig = plot_pred_vs_true_dimension_reduction_results(
                    targets=comp_emb,
                    predictions=embeddings,
                    dimension_reduction_model=model,
                    dimension_reduction_model_name=name,
                    style=style,
                )
                safe_write_image(fig, plots_path / f"{name}.png")
        except FileNotFoundError:
            logger.warning("Comparison embeddings not found; skipping comparison plot")

    def validate(self, model_config: OmegaConf, **kwargs: Any) -> None:
        embeddings, targets = self.load_embeddings_and_targets(
            model_config=model_config,
            **filter_kwargs(self.load_embeddings_and_targets, kwargs),
        )
        kwargs["embeddings"] = embeddings

        target_indices = getattr(self.dataset, "target_indices", None) or self.head_config.get(
            "target_indices", None,
        )
        if target_indices is not None:
            targets = self._maybe_subset_targets(targets, list(target_indices))
            self.targets_names = get_filtered_targets_names(self.dataset_name, list(target_indices))

        if targets.ndim == 1:
            is_binary = True
            targets = targets.reshape(-1, 1)
        elif targets.shape[1] == 1:
            is_binary = True
        else:
            is_binary = False
        kwargs["is_binary"] = is_binary

        metrics_per_run: list[dict[str, Any]] = []

        for i in range(self.n_runs):
            run_id = self.start_run_id + i
            logger.info("Run %s started", run_id)
            x_train, x_test, y_train, y_test = self.prepare_data(
                targets=targets,
                random_seed=self.random_seed + run_id,
                **filter_kwargs(self.prepare_data, kwargs),
            )

            # All heads share a 72/8/20 train/validation/test split; heads
            # that do not use validation data ignore it.
            x_train_h, x_val_h, y_train_h, y_val_h = train_test_split(
                x_train, y_train, test_size=0.1, random_state=self.random_seed + run_id,
            )
            head = self.create_head()
            # Thread the per-run seed into stochastic heads for seed-to-seed variance.
            # XGBoost's seed lives at head_config.params.seed (not top-level); its per-run
            # variance comes from the data split, which is sufficient for Wilcoxon.
            if hasattr(head, "head_config") and "seed" in head.head_config:
                head.head_config.seed = self.random_seed + run_id
            else:
                logger.debug(
                    "Head %s has no top-level seed; per-run seed not threaded "
                    "(variance from split randomness).",
                    type(head).__name__,
                )
            fit_kwargs = self.get_fit_kwargs(x_train_h, x_val_h, y_train_h, y_val_h)
            try:
                head.fit(x_train_h, y_train_h, x_val=x_val_h, y_val=y_val_h, **fit_kwargs)
            except NotImplementedError as exc:
                logger.warning(
                    "Head %s raised NotImplementedError on fit (sample_weight unsupported); "
                    "omitting run %s: %s",
                    type(head).__name__,
                    run_id,
                    exc,
                )
                continue
            except RuntimeError as exc:
                logger.error(
                    "Head %s raised RuntimeError on fit; omitting run %s: %s",
                    type(head).__name__,
                    run_id,
                    exc,
                )
                continue

            predicted = head.predict(x_test)
            if self.head_config.get("protocol") == "uniform":
                # Record validation scores for selection using the fitted head;
                # validation data are not used for fitting.
                self._save_validation_metrics(model_config, head, x_val_h, y_val_h, run_id, is_binary, kwargs)
            additional = self.get_additional_predictions(
                head,
                x_test,
                **filter_kwargs(self.get_additional_predictions, kwargs),
            )
            if is_binary:
                predicted = predicted.reshape(-1, 1) if predicted.ndim == 1 else predicted

            metrics = self.calculate_metrics(
                y_test,
                predicted,
                **additional,
                **filter_kwargs(self.calculate_metrics, kwargs),
            )
            metrics_per_run.append(metrics)
            self.save_metrics_for_run(model_config=model_config, metrics=metrics, run_id=run_id)
            self.save_data_for_run(model_config, y_test, predicted, run_id)
            self.save_head_for_run(model_config, head, run_id)
            self.plot_results_for_run(
                model_config=model_config,
                targets=y_test,
                predictions=predicted,
                metrics=metrics,
                run_id=run_id,
                **filter_kwargs(self.plot_results_for_run, kwargs),
            )
            logger.info("Run %s done", run_id)

        if not metrics_per_run:
            raise RuntimeError(
                f"All {self.n_runs} runs were skipped for head "
                f"{self.head_config.get('head_type', '?')} on dataset {self.dataset_name}. "
                "Check head compatibility with this task (e.g. sample_weight support).",
            )
        # When n_runs == 1 (the per-seed grid layout), "global" stats over a single
        # sample are meaningless and every task would clobber metrics/global/ with that
        # task's lone value. Skip in that case — proper cross-seed aggregation is the
        # job of an offline aggregator that reads all metrics/<seed>/ dirs.
        if self.n_runs > 1:
            global_metrics, errors = self.get_global_metrics_and_errors(metrics_per_run)
            self.save_metrics_global(model_config, global_metrics, errors)
            self.plot_results_global(
                model_config=model_config,
                global_metrics=global_metrics,
                **filter_kwargs(self.plot_results_global, kwargs),
            )
        logger.info("Validation done")

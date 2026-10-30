"""Validator that uses EGL for uncertainty estimation.

## EGL
EGL (Expected Gradient Length) is a method for uncertainty estimation.
It's computed by running a forward pass, measuring loss between output and
pseudo-output (output + noise) as true value. Then, backpropagating the loss
and calculating the norm of the gradient.
The higher the EGL, the more uncertain the prediction is.
"""

import json
import logging
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L  # noqa: N812
import numpy as np
import torch
from omegaconf import OmegaConf
from plotly import graph_objects as go
from rich.console import Console
from rich.table import Table
from scipy.stats import wasserstein_distance
from torch import nn
from tqdm.auto import tqdm, trange

from utils.load_models import load_model
from utils.minor import put_on_device, safe_write_image
from utils.model_display import create_pretty_name
from utils.typing import DataItem
from validation.uncertainty.abstract import UncertaintyValidator
from validation.uncertainty.compare_with_test import CompareWithTestValidator
from validation.utils import get_collate
from vis.pieces import plot_boxplot, plot_histogram

logger = logging.getLogger(__name__)

NEW_MODULE_PARAMS_MSG = (
    "Module parameters changed while BatchNormFreezer was in use. Parameters"
    " {param_names} are new and was not expected."
)

PARAM_NAME_MISSING_MSG = (
    "Module parameters changed while BatchNormFreezer was in use. Parameters"
    " {param_names} are missing."
)


def freeze_batch_norms(model: nn.Module | L.LightningModule) -> None:
    """Freeze all batch normalization layers in the model.

    This function is used when we need to compute some statistics about the model
    in training mode, but not during training. In such cases we do not want
    the batch normalization layers to adapt to the inputs we pass to them.

    Parameters
    ----------
    model : nn.Module | L.LightningModule
        The model whose batch normalization layers should be frozen.

    """
    for module in model.modules():
        if isinstance(
            module,
            nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d | nn.LayerNorm,
        ):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            module.track_running_stats = False


def turn_off_dropouts(model: nn.Module | L.LightningModule) -> None:
    """Turns off dropout layers in a model.

    This function is used when we need to compute some statistics about the model
    in training mode, but not during training. In such cases we do not want
    the dropout layers to drop out neurons.

    Parameters
    ----------
    model : nn.Module | L.LightningModule
        The model whose dropout layers should be turned off.

    Examples
    --------
    >>> def compute_egls(model, dataset):
    ...     \"\"\"Some function that computes EGLs.\"\"\"
    ...     ...
    >>> model = MyModel()
    >>> with BatchNormFreezer(model):
    >>>     turn_off_dropouts(model)
    >>>     egls = compute_egls(model, dataset)

    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


class BatchNormFreezer:
    """Context manager for freezing batch normalization layers in a model.

    Freezing batch normalization layers is required when we compute some statistics
    about model in training mode, but not during training. In such cases we do not want
    the batch normalization layers to adapt to the inputs we pass to them.

    Parameters
    ----------
    model : nn.Module | L.LightningModule
        The model whose batch normalization layers should be frozen.

    Examples
    --------
    >>> def compute_egls(model, dataset):
    ...     \"\"\"Some function that computes EGLs.\"\"\"
    ...     ...
    >>> model = MyModel()
    >>> with BatchNormFreezer(model):
    >>>     egls = compute_egls(model, dataset)

    """

    def __init__(self, model: nn.Module | L.LightningModule) -> None:
        """Initializes a BatchNormFreezer.

        Parameters
        ----------
        model : nn.Module | L.LightningModule
            The model whose batch normalization layers should be frozen.

        """
        self.model = model
        self.previous_states = []

    def __enter__(self) -> None:
        """Freezes all batch normalization layers in the model.

        When the context manager is entered, it saves the current state of all
        batch normalization layers and sets them to evaluation mode, disables their
        gradients and stops tracking running statistics. This is useful when
        computing uncertainty estimates, as it prevents the batch normalization
        layers from adapting to the noise in the inputs.

        When the context manager is exited, it restores the state of the batch
        normalization layers to their original values.

        Returns:
            None

        """
        for module in self.model.modules():
            if isinstance(
                module,
                nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d | nn.LayerNorm,
            ):
                self.previous_states.append(
                    {
                        "module": module,
                        "training": module.training,
                        "requires_grads": {
                            name: param.requires_grad for name, param in module.named_parameters()
                        },
                        "track_running_stats": module.track_running_stats,
                    },
                )
        freeze_batch_norms(self.model)

    def __exit__(self, *args: object, **kwargs: dict) -> None:
        """Restores the state of all batch normalization layers in the model.

        When the context manager is exited, it restores the state of the batch
        normalization layers to their original values.

        Returns:
            None

        """
        for state in self.previous_states:
            if state["training"]:
                state["module"].train()
                not_found = set(state["requires_grads"].keys()) - set(
                    state["module"].named_parameters().keys(),
                )
                if not_found:
                    raise RuntimeError(
                        PARAM_NAME_MISSING_MSG.format(param_name=list(not_found)),
                    )
                new = set(state["module"].named_parameters().keys()) - set(
                    state["requires_grads"].keys(),
                )
                if new:
                    raise RuntimeError(
                        NEW_MODULE_PARAMS_MSG.format(param_names=list(not_found)),
                    )

                for param_name, param in state["module"].named_parameters():
                    param.requires_grad = state["requires_grads"][param_name]

            state["module"].track_running_stats = state["track_running_stats"]
        self.previous_states = []


UNKNOWN_PARALLELISM_METHOD = "Unknown parallelism method: {parallelism_method}"


class EGLValidator(UncertaintyValidator):
    """Validator that uses EGL for uncertainty estimation.

    ## EGL
    EGL (Expected Gradient Length) is a method for uncertainty estimation.
    It's computed by running a forward pass, measuring loss between output and
    pseudo-output (output + noise) as true value. Then, backpropagating the loss
    and calculating the norm of the gradient.
    The higher the EGL, the more uncertain the prediction is.

    Parameters
    ----------
    dataset : Iterable[torch.Tensor | torch_geometric.data.Data]
        The dataset to evaluate the uncertainty on.
    dataset_name : str
        The name of the dataset.
    noise_scale : float, optional
        The scale of the noise to add to the inputs, by default 0.01.
    loss : nn.Module | Literal["l1", "mae", "l2", "mse"], optional
        The loss function to use, by default nn.L1Loss(reduction="mean").

    """

    ITERS_BETWEEN_UPDATES = 2

    def __init__(
        self,
        dataset: Iterable[DataItem],
        dataset_name: str,
        noise_scale: float = 0.01,
        loss_or_creator: type[nn.Module] | Literal["l1", "mae", "l2", "mse"] = "l1",
    ) -> None:
        """Initializes an EGLValidator object.

        Parameters
        ----------
        dataset : Iterable[torch.Tensor | torch_geometric.data.Data]
            The dataset to evaluate the uncertainty on.
        dataset_name : str
            The name of the dataset.
        noise_scale : float, optional
            The scale of the noise to add to the inputs, by default 0.01.
        loss : nn.Module | Literal["l1", "mae", "l2", "mse"], optional
            The loss function to use, by default nn.L1Loss(reduction="mean").

        """
        if isinstance(loss_or_creator, str):
            loss_or_creator = {
                "l1": partial(nn.L1Loss, reduction="mean"),
                "mae": partial(nn.L1Loss, reduction="mean"),
                "l2": partial(nn.MSELoss, reduction="mean"),
                "mse": partial(nn.MSELoss, reduction="mean"),
            }[loss_or_creator]
        self.loss_creator = loss_or_creator
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.noise_scale = noise_scale

    def run_validation(
        self,
        model_config: OmegaConf,
        chunk_size: int = 16,
        max_items_on_device: int = 10000,
        parallelism_method: Literal["cpu", "cuda"] | None = None,
        *,
        recalculate: bool = False,
    ) -> list[float]:
        """Calculate the uncertainty of the model by computing the EGLs.

        Parameters
        ----------
        model_config : OmegaConf
            The configuration of the model.
        chunk_size : int
            The number of items to process in each chunk.
        recalculate : bool
            Whether to recalculate the EGLs.

        Returns
        -------
        dict[str, list[float]]:
            A dict with the following keys:
                - `scores`: A list of the gradient lengths calculated for each item in the dataset.

        """
        collate_fn = get_collate(self.dataset)

        try:
            scores = self._try_load_scores(
                model_config=model_config,
                recalculate=recalculate,
            )
        except FileNotFoundError:
            scores = self._run_validation_core(
                model_config=model_config,
                collate_fn=collate_fn,
                chunk_size=chunk_size,
                max_items_on_device=max_items_on_device,
                parallelism_method=parallelism_method,
            )

        mean_egl = np.mean(scores)

        logger.info("Mean EGL: %s", mean_egl)
        median_egl = np.median(scores)
        logger.info("Median EGL: %s", median_egl)
        std = np.std(scores)
        logger.info("Std: %s", std)

        return {"scores": scores, "mean_egl": mean_egl, "median_egl": median_egl, "std": std}

    def _try_load_scores(
        self,
        model_config: OmegaConf,
        *,
        recalculate: bool = False,
    ) -> list[float]:
        if recalculate:
            raise FileNotFoundError
        save_path = Path("results") / self.dataset_name / model_config.name / "egl"
        return np.load(save_path / "scores.npy")

    def _run_validation_core(
        self,
        model_config: OmegaConf,
        collate_fn: Callable[[list[DataItem]], DataItem],
        chunk_size: int,
        max_items_on_device: int,
        parallelism_method: Literal["cpu", "cuda"] | None,
    ) -> None:
        if parallelism_method is None:
            batch_size_one_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
            )
            logger.info("Loading model")
            model = load_model(model_config)
            logger.info("Preparing model")
            model.mode = "embedding"
            example_chunk = next(iter(batch_size_one_loader))
            noise_shape, noise_dtype = self._get_noise_shape_and_dtype(
                model,
                example_chunk,
            )
            model.train()  # enable gradients
            freeze_batch_norms(model)
            turn_off_dropouts(model)
            loss = self.loss_creator()
            logger.info("Running EGL calculation")
            scores = self._run_validation_core_no_parallel(
                model=model,
                batch_size_one_loader=batch_size_one_loader,
                loss=loss,
                noise_shape=noise_shape,
                noise_dtype=noise_dtype,
            )
        else:
            loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=chunk_size,
                shuffle=False,
                collate_fn=lambda x: x,
            )
            models = [
                load_model(model_config)
                for _ in (
                    trange(chunk_size, desc="Loading models")
                    if logger.isEnabledFor(logging.INFO)
                    else range(chunk_size)
                )
            ]
            for model in models:
                model.mode = "embedding"
            with torch.no_grad():
                example_chunk = next(iter(loader))
                noise_shape, noise_dtype = self._get_noise_shape_and_dtype(
                    models[0],
                    collate_fn([example_chunk]),
                )
            for model in models:
                model.train()  # enable gradients
                freeze_batch_norms(model)
                turn_off_dropouts(model)
            losses = [self.loss_creator() for _ in (range(chunk_size))]
            if parallelism_method == "cuda":
                scores = self._run_validation_core_cuda(
                    models=models,
                    loader=loader,
                    chunk_size=chunk_size,
                    collate_fn=collate_fn,
                    max_items_on_device=max_items_on_device,
                    losses=losses,
                    noise_shape=noise_shape,
                    noise_dtype=noise_dtype,
                )
            elif parallelism_method == "cpu":
                scores = self._run_validation_core_cpu(
                    models=models,
                    loader=loader,
                    collate_fn=collate_fn,
                    losses=losses,
                    noise_shape=noise_shape,
                    noise_dtype=noise_dtype,
                )
            else:
                raise ValueError(
                    UNKNOWN_PARALLELISM_METHOD.format(parallelism_method=parallelism_method),
                )

        return scores

    def _get_noise_shape_and_dtype(
        self,
        model: nn.Module | L.LightningModule,
        data: DataItem,
    ) -> tuple[tuple[int, ...], torch.dtype]:
        with torch.inference_mode():
            y_pred = model(put_on_device(model, data))
        noise_shape = y_pred.shape
        noise_dtype = y_pred.dtype
        return noise_shape, noise_dtype

    def _run_validation_core_no_parallel(
        self,
        model: nn.Module | L.LightningModule,
        batch_size_one_loader: torch.utils.data.DataLoader,
        loss: nn.Module,
        noise_shape: tuple[int, ...] = (1,),
        noise_dtype: torch.dtype = torch.float32,
    ) -> list[float]:
        progress_bar = (
            tqdm(batch_size_one_loader, desc="Computing EGLs", unit="molecule")
            if logger.isEnabledFor(logging.INFO)
            else batch_size_one_loader
        )
        return [
            _compute_one_egl(
                model=model,
                data=put_on_device(model, data),
                loss=loss,
                noise=torch.randn(noise_shape, dtype=noise_dtype, device=model.device),
            )
            for data in progress_bar
        ]

    def _run_validation_core_cuda(  # noqa: PLR0913, C901
        self,
        models: list[nn.Module | L.LightningModule],
        loader: torch.utils.data.DataLoader,
        chunk_size: int,
        collate_fn: Callable[[list[DataItem]], DataItem],
        max_items_on_device: int,
        losses: list[nn.Module],
        noise_shape: tuple[int, ...],
        noise_dtype: torch.dtype,
    ) -> None:
        device = "cuda"
        progress_bar = (
            tqdm(total=len(self.dataset) // chunk_size, desc="Computing EGLs", unit="chunk")
            if logger.isEnabledFor(logging.INFO)
            else None
        )
        for model in models:
            model.to(device)
        mean_egl = 0
        n_egls = 0
        remaining = True
        iterator = iter(loader)
        on_device = []
        scores = []
        streams = [
            torch.cuda.Stream()
            for _ in (
                trange(chunk_size, desc="Creating CUDA streams")
                if logger.isEnabledFor(logging.INFO)
                else range(chunk_size)
            )
        ]

        while remaining:
            try:
                new_off_device = next(iterator)
                new_off_device = [collate_fn([item]) for item in new_off_device]
                new_on_device = [put_on_device(models[0], item) for item in new_off_device]
                on_device.append(new_on_device)
            except StopIteration:
                remaining = False

            if not remaining or len(on_device) * chunk_size > max_items_on_device:
                for i, chunk in enumerate(on_device):
                    these_scores = [None] * chunk_size
                    for j, (data, model, stream, loss) in enumerate(
                        zip(chunk, models, streams, losses, strict=False),
                    ):
                        with torch.cuda.stream(stream):
                            grad_length = _compute_one_egl(
                                model,
                                data,
                                loss,
                                self.noise_scale
                                * torch.randn(
                                    noise_shape,
                                    dtype=noise_dtype,
                                    device=device,
                                ),
                            )
                            these_scores[j] = grad_length
                    torch.cuda.synchronize()
                    score_sum = 0
                    score_n = 0
                    for score in these_scores:
                        if score is not None:
                            score_sum += score
                            score_n += 1
                            scores.append(score)
                    mean_egl = (mean_egl * n_egls + score_sum) / (n_egls + score_n)
                    n_egls += score_n

                    if progress_bar is not None:
                        if i % self.ITERS_BETWEEN_UPDATES == 0:
                            progress_bar.set_postfix(mean_egl=mean_egl)
                        progress_bar.update()
        return scores, mean_egl

    def _run_validation_core_cpu(
        self,
        models: list[nn.Module | L.LightningModule],
        loader: torch.utils.data.DataLoader,
        collate_fn: Callable[[list[DataItem]], DataItem],
        losses: list[nn.Module],
        noise_shape: tuple[int, ...],
        noise_dtype: torch.dtype,
    ) -> None:
        models = [model.cpu() for model in models]
        progress_bar = (
            tqdm(loader, desc="Computing EGLs", unit="chunk")
            if logger.isEnabledFor(logging.INFO)
            else loader
        )
        scores = []
        logger.info("Computing EGLs")
        for i, chunk_unbatched in enumerate(progress_bar):
            chunk = [collate_fn([item]) for item in chunk_unbatched]

            args_list = [
                (
                    model,
                    data,
                    model_idx,
                    loss,
                    self.noise_scale * torch.randn(noise_shape, dtype=noise_dtype),
                    logger.isEnabledFor(logging.DEBUG),
                )
                for model_idx, (model, data, loss) in enumerate(
                    zip(models, chunk, losses, strict=False),
                )
            ]
            logger.debug("Starting pool")
            with torch.multiprocessing.Pool() as pool:
                results = pool.map(_compute_one_egl_wrapped_for_pool, args_list)

            scores.extend([res["score"] for res in results])

            if i % self.ITERS_BETWEEN_UPDATES == 0 and progress_bar is not None:
                progress_bar.set_postfix(mean_egl=np.mean(scores))
        return scores

    def save_results(self, results: dict, model_config: OmegaConf) -> None:
        """Saves the results of the EGL validation.

        Parameters
        ----------
        results : dict
            A dictionary containing the results of the EGL validation.
        model_config : OmegaConf
            The configuration of the model.

        Returns
        -------
        None

        """
        save_path = Path("results") / self.dataset_name / model_config.name / "egl"
        save_path.mkdir(parents=True, exist_ok=True)
        scores_npy = np.array(results["scores"])
        np.save(save_path / "scores.npy", scores_npy)
        table = Table("Mean", "Median", "Std", title="EGLs")
        table.add_row(
            f"{results['mean_egl']:.4f}",
            f"{results['median_egl']:.4f}",
            f"{results['std']:.4f}",
        )
        if logger.isEnabledFor(logging.INFO):
            console = Console()
            console.print(table)

        with (save_path / "report.txt").open("w", encoding="utf-8") as file:
            console = Console(file=file)
            console.print(table)

    def plot_results(self, results: dict, model_config: OmegaConf) -> None:
        """Plot the results of the EGL validation.

        Plots are saved in `results/{dataset_name}/{model_name}/egl`.

        Parameters
        ----------
        results : dict
            A dictionary containing the results of the EGL validation.
        model_config : OmegaConf
            The configuration of the model.

        Returns
        -------
        None

        """
        save_path = Path("results") / self.dataset_name / model_config.name / "egl"

        logger.info("Plotting EGL results")
        logger.info("Plotting scores box plot")
        fig = go.Figure()
        fig = plot_boxplot(fig, results["scores"], "EGL")
        pretty_model_name = create_pretty_name(
            model_config=model_config,
            model_name=model_config.name,
        )
        fig.update_layout(title=f"EGL histogram for {pretty_model_name}")
        safe_write_image(fig, save_path / "scores_box.png")

        logger.info("Plotting scores histogram")
        fig = go.Figure()
        fig = plot_histogram(fig, results["scores"], "EGL")
        safe_write_image(fig, save_path / "scores_histogram.png")
        logger.info("Plotting scores histogram normalized")
        fig = go.Figure()
        fig = plot_histogram(
            fig,
            results["scores"],
            "EGL",
            histogram_kwargs={"histnorm": "percent"},
        )
        fig.update_layout(title=f"EGL histogram for {pretty_model_name}")
        safe_write_image(fig, save_path / "scores_histogram_normalized.png")


def _compute_one_egl_wrapped_for_pool(args: tuple) -> float:
    model, data, model_idx, loss, noise, debug = args
    return {
        "model_idx": model_idx,
        "score": _compute_one_egl(model=model, data=data, loss=loss, noise=noise),
    }


def _compute_one_egl(
    model: nn.Module | L.LightningModule,
    data: DataItem,
    loss: nn.Module,
    noise: torch.Tensor,
) -> None:
    y_pred = model(data)
    pseudo_y = y_pred.detach() + noise
    loss = loss(y_pred, pseudo_y)
    model.zero_grad()
    loss.backward()

    grad_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_sq += param.grad.norm().item() ** 2
    grad_length = grad_norm_sq**0.5

    return grad_length  # noqa:RET504


class CompareEGLWithTestValidator(CompareWithTestValidator, validator_class=EGLValidator):
    def run_comparison(
        self,
        model_config: OmegaConf,  # noqa: ARG002
        anchor_results: dict,
        reference_results: dict,
    ) -> dict:
        """Runs the comparison of the anchor and reference validators.

        Parameters
        ----------
        model_config : OmegaConf
            The model configuration.
        anchor_results : dict
            The results of the anchor validator.
        reference_results : dict
            The results of the reference validator.

        Returns
        -------
        dict
            A dictionary containing the comparison results.

        """
        means_diff = anchor_results["mean_egl"] - reference_results["mean_egl"]
        medians_diff = anchor_results["median_egl"] - reference_results["median_egl"]
        egl_wasserstein_distance = wasserstein_distance(
            anchor_results["scores"],
            reference_results["scores"],
        )

        return {
            "means_diff": means_diff,
            "medians_diff": medians_diff,
            "egl_wasserstein_distance": egl_wasserstein_distance,
            "anchor_results": anchor_results,
            "reference_results": reference_results,
        }

    def save_results(self, results: dict, model_config: OmegaConf) -> None:
        """Saves the results of the comparison of the anchor and reference validators.

        Parameters
        ----------
        results : dict
            The results of the comparison.
        model_config : OmegaConf
            The model configuration.

        Returns
        -------
        None

        """
        save_path = Path("results") / self.dataset_name / model_config.name / "egl_comparison"
        save_path.mkdir(parents=True, exist_ok=True)
        table = Table("ΔMean EGL", "ΔMedian EGL", "Wasserstein Distance")
        table.add_row(
            f"{results['means_diff']:.4f}",
            f"{results['medians_diff']:.4f}",
            f"{results['egl_wasserstein_distance']:.4f}",
        )

        if logger.isEnabledFor(logging.INFO):
            console = Console()
            console.print(table)

        with (save_path / "report.txt").open("w", encoding="utf-8") as file:
            console = Console(file=file)
            console.print(table)

        with (save_path / "report.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "means_diff": results["means_diff"],
                    "medians_diff": results["medians_diff"],
                    "egl_wasserstein_distance": results["egl_wasserstein_distance"],
                },
                file,
                indent=4,
            )

    def plot_results(self, results: dict, model_config: OmegaConf) -> None:
        fig = go.Figure()
        fig = plot_histogram(
            fig,
            results["anchor_results"]["scores"],
            self.anchor_validator.dataset_name,
        )
        fig = plot_histogram(
            fig,
            results["reference_results"]["scores"],
            self.reference_validator.dataset_name,
        )
        pretty_model_name = create_pretty_name(
            model_config=model_config,
            model_name=model_config.name,
        )
        fig.update_layout(title=f"EGL histograms for {pretty_model_name}")
        safe_write_image(
            fig,
            Path("results")
            / self.dataset_name
            / model_config.name
            / "egl_comparison"
            / "scores_histogram.png",
        )
        fig = go.Figure()
        fig = plot_histogram(
            fig,
            results["anchor_results"]["scores"],
            self.anchor_validator.dataset_name,
            histogram_kwargs={"histnorm": "percent"},
        )
        fig = plot_histogram(
            fig,
            results["reference_results"]["scores"],
            self.reference_validator.dataset_name,
            histogram_kwargs={"histnorm": "percent"},
        )
        fig.update_layout(title=f"EGL histograms for {pretty_model_name}")
        safe_write_image(
            fig,
            Path("results")
            / self.dataset_name
            / model_config.name
            / "egl_comparison"
            / "scores_histogram.png",
        )

"""Tests for all Head implementations: XGBoost, Linear, MLP, GP-*, TabPFN, TabICL."""

import os

import numpy as np
import pytest
from omegaconf import OmegaConf

import validation.heads.linear  # noqa: F401
from validation.heads import load_head_from_config
from validation.heads.xgboost import XGBoostHead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_regression():
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((100, 10)).astype(np.float32)
    y_train = rng.standard_normal(100).astype(np.float32)
    x_test = rng.standard_normal((20, 10)).astype(np.float32)
    y_test = rng.standard_normal(20).astype(np.float32)
    return x_train, y_train, x_test, y_test


@pytest.fixture()
def synthetic_classification():
    rng = np.random.default_rng(1)
    x_train = rng.standard_normal((100, 10)).astype(np.float32)
    # Multi-column binary labels as expected by calculate_sample_weights
    y_train = rng.integers(0, 2, size=(100, 2)).astype(np.float32)
    x_test = rng.standard_normal((20, 10)).astype(np.float32)
    y_test = rng.integers(0, 2, size=(20, 2)).astype(np.float32)
    return x_train, y_train, x_test, y_test


def _make_reg_cfg(**extra_params):
    params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.3, "device": "cpu", "seed": 42}
    params.update(extra_params)
    return OmegaConf.create({
        "head_type": "xgboost",
        "objective_type": "regression",
        "path": "/tmp/xgb_test_reg.json",
        "num_boost_round": 10,
        "params": params,
    })


def _make_cls_cfg(**extra_params):
    params = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3, "device": "cpu", "seed": 42}
    params.update(extra_params)
    return OmegaConf.create({
        "head_type": "xgboost",
        "objective_type": "classification",
        "path": "/tmp/xgb_test_cls.json",
        "num_boost_round": 10,
        "params": params,
    })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_xgboost_head_registered() -> None:
    from validation.heads import HEAD_REGISTRY
    assert "xgboost" in HEAD_REGISTRY


def test_load_head_from_config_returns_xgboost_head(synthetic_regression) -> None:
    cfg = _make_reg_cfg()
    head = load_head_from_config(cfg)
    assert isinstance(head, XGBoostHead)


# ---------------------------------------------------------------------------
# Fix 1: eval_set only when early_stopping_rounds is set
# ---------------------------------------------------------------------------

def test_no_eval_set_without_early_stopping(synthetic_regression, monkeypatch) -> None:
    """Without early_stopping_rounds, fit() must use full training data (no split)."""
    x_train, y_train, x_test, _ = synthetic_regression
    cfg = _make_reg_cfg()  # no early_stopping_rounds
    head = load_head_from_config(cfg)

    fit_calls = []
    original_fit = head._make_xgb_model

    def patched_make_model():
        model = original_fit()
        original_model_fit = model.fit

        def recording_fit(*args, **kwargs):
            fit_calls.append({"args": args, "kwargs": kwargs})
            return original_model_fit(*args, **kwargs)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(head, "_make_xgb_model", patched_make_model)
    head.fit(x_train, y_train)

    assert len(fit_calls) == 1
    assert "eval_set" not in fit_calls[0]["kwargs"], "eval_set should not be passed without early_stopping_rounds"
    # Full dataset should be used (no split)
    assert fit_calls[0]["args"][0].shape[0] == x_train.shape[0]


def test_eval_set_passed_with_early_stopping(synthetic_regression, monkeypatch) -> None:
    """With early_stopping_rounds set, fit() must auto-split and pass eval_set."""
    x_train, y_train, x_test, _ = synthetic_regression
    cfg = _make_reg_cfg(early_stopping_rounds=5)
    head = load_head_from_config(cfg)

    fit_calls = []
    original_make = head._make_xgb_model

    def patched_make_model():
        model = original_make()
        orig_fit = model.fit

        def recording_fit(*args, **kwargs):
            fit_calls.append({"args": args, "kwargs": kwargs})
            return orig_fit(*args, **kwargs)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(head, "_make_xgb_model", patched_make_model)
    head.fit(x_train, y_train)

    assert len(fit_calls) == 1
    assert "eval_set" in fit_calls[0]["kwargs"], "eval_set should be passed when early_stopping_rounds is set"
    # 10% split → training set should be ~90 samples
    n_train_used = fit_calls[0]["args"][0].shape[0]
    assert n_train_used < x_train.shape[0], "Training data should be split when early_stopping_rounds is set"


# ---------------------------------------------------------------------------
# Fix 2: Classification auto-computes sample weights
# ---------------------------------------------------------------------------

def test_classification_auto_sample_weights(synthetic_classification, monkeypatch) -> None:
    """When task=classification and no sample_weight given, weights are auto-computed."""
    x_train, y_train, x_test, _ = synthetic_classification
    cfg = _make_cls_cfg()
    head = load_head_from_config(cfg)

    fit_calls = []
    original_make = head._make_xgb_model

    def patched_make_model():
        model = original_make()
        orig_fit = model.fit

        def recording_fit(*args, **kwargs):
            fit_calls.append({"args": args, "kwargs": kwargs})
            return orig_fit(*args, **kwargs)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(head, "_make_xgb_model", patched_make_model)
    head.fit(x_train, y_train)

    assert len(fit_calls) == 1
    assert "sample_weight" in fit_calls[0]["kwargs"], "sample_weight should be auto-computed for classification"
    sw = fit_calls[0]["kwargs"]["sample_weight"]
    assert sw is not None
    assert len(sw) == x_train.shape[0]


def test_regression_no_auto_sample_weights(synthetic_regression, monkeypatch) -> None:
    """For regression, no auto sample weights should be applied."""
    x_train, y_train, x_test, _ = synthetic_regression
    cfg = _make_reg_cfg()
    head = load_head_from_config(cfg)

    fit_calls = []
    original_make = head._make_xgb_model

    def patched_make_model():
        model = original_make()
        orig_fit = model.fit

        def recording_fit(*args, **kwargs):
            fit_calls.append({"args": args, "kwargs": kwargs})
            return orig_fit(*args, **kwargs)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(head, "_make_xgb_model", patched_make_model)
    head.fit(x_train, y_train)

    assert "sample_weight" not in fit_calls[0]["kwargs"], "regression should not auto-compute sample_weight"


def test_explicit_sample_weight_not_overridden(synthetic_classification) -> None:
    """When sample_weight is explicitly passed, it should not be replaced."""
    x_train, y_train, x_test, _ = synthetic_classification
    cfg = _make_cls_cfg()
    head = load_head_from_config(cfg)

    explicit_weight = np.ones(x_train.shape[0], dtype=np.float32)
    # Should not raise; the explicit weight should be used
    head.fit(x_train, y_train, sample_weight=explicit_weight)
    assert head._model is not None


# ---------------------------------------------------------------------------
# Fix 3: predict / predict_proba return np.ndarray
# ---------------------------------------------------------------------------

def test_predict_returns_ndarray(synthetic_regression) -> None:
    x_train, y_train, x_test, _ = synthetic_regression
    cfg = _make_reg_cfg()
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train)
    result = head.predict(x_test)
    assert isinstance(result, np.ndarray), f"predict should return np.ndarray, got {type(result)}"
    assert result.shape == (x_test.shape[0],)


def test_predict_proba_returns_ndarray(synthetic_classification) -> None:
    x_train, y_train, x_test, _ = synthetic_classification
    cfg = _make_cls_cfg()
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train)
    result = head.predict_proba(x_test)
    assert isinstance(result, np.ndarray), f"predict_proba should return np.ndarray, got {type(result)}"
    assert result.shape[0] == x_test.shape[0]


# ---------------------------------------------------------------------------
# Fix 4: predict_proba raises on unfit classifier
# ---------------------------------------------------------------------------

def test_predict_proba_raises_before_fit() -> None:
    cfg = _make_cls_cfg()
    head = load_head_from_config(cfg)
    x_test = np.random.default_rng(0).standard_normal((10, 10)).astype(np.float32)
    with pytest.raises(RuntimeError, match="called before fit"):
        head.predict_proba(x_test)


def test_predict_raises_before_fit() -> None:
    cfg = _make_reg_cfg()
    head = load_head_from_config(cfg)
    x_test = np.random.default_rng(0).standard_normal((10, 10)).astype(np.float32)
    with pytest.raises(RuntimeError, match="called before fit"):
        head.predict(x_test)


def test_save_raises_before_fit(tmp_path) -> None:
    cfg = _make_reg_cfg()
    head = load_head_from_config(cfg)
    with pytest.raises(RuntimeError, match="called before fit"):
        head.save(tmp_path / "model.json")


# ---------------------------------------------------------------------------
# predict_proba returns None for regression
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LinearHead fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_regression_linear():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((200, 5)).astype(np.float32)
    w = np.array([2.0, -1.5, 0.8, -0.3, 1.2], dtype=np.float32)
    y = x @ w + 0.1 * rng.standard_normal(200).astype(np.float32)
    return x[:150], y[:150], x[150:], y[150:]


@pytest.fixture()
def synthetic_binary_classification():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((200, 5)).astype(np.float32)
    labels = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x[:150], labels[:150], x[150:], labels[150:]


# ---------------------------------------------------------------------------
# LinearHead tests
# ---------------------------------------------------------------------------

def test_linear_head_regression_fits_and_predicts(synthetic_regression_linear) -> None:
    x_train, y_train, x_test, y_test = synthetic_regression_linear
    cfg = OmegaConf.create({
        "head_type": "linear",
        "objective_type": "regression",
        "alphas": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "cv": 5,
    })
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train)
    pred = head.predict(x_test)
    r2 = 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    assert r2 > 0.5


def test_linear_head_classification_predict_proba(synthetic_binary_classification) -> None:
    x_train, y_train, x_test, _ = synthetic_binary_classification
    cfg = OmegaConf.create({
        "head_type": "linear",
        "objective_type": "classification",
        "Cs": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "cv": 5,
        "max_iter": 2000,
    })
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train)
    proba = head.predict_proba(x_test)
    assert proba is not None
    assert proba.shape == (x_test.shape[0], 2)


# ---------------------------------------------------------------------------
# XGBoostHead — predict_proba returns None for regression
# ---------------------------------------------------------------------------

import validation.heads.mlp  # noqa: F401


def test_mlp_head_regression_fits_and_predicts(synthetic_regression_linear) -> None:
    x_train, y_train, x_test, y_test = synthetic_regression_linear
    cfg = OmegaConf.create(
        {
            "head_type": "mlp",
            "objective_type": "regression",
            "hidden_dim": 64,
            "n_hidden_layers": 2,
            "activation": "gelu",
            "dropout": 0.0,
            "lr": 5.0e-3,
            "weight_decay": 0.0,
            "batch_size": 32,
            "max_epochs": 100,
            "val_fraction": 0.1,
            "early_stopping_patience": 20,
            "seed": 42,
        },
    )
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train)
    pred = head.predict(x_test)
    assert pred.shape == (x_test.shape[0],)
    r2 = 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    assert r2 > 0.5


def test_mlp_head_supports_multi_target_regression() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((200, 16)).astype(np.float32)
    w = rng.standard_normal((16, 3)).astype(np.float32)
    y = (x @ w + 0.1 * rng.standard_normal((200, 3))).astype(np.float32)
    cfg = OmegaConf.create(
        {
            "head_type": "mlp",
            "objective_type": "regression",
            "hidden_dim": 64,
            "n_hidden_layers": 2,
            "activation": "gelu",
            "dropout": 0.0,
            "lr": 5.0e-3,
            "weight_decay": 0.0,
            "batch_size": 32,
            "max_epochs": 30,
            "val_fraction": 0.1,
            "early_stopping_patience": 5,
            "seed": 0,
        },
    )
    head = load_head_from_config(cfg)
    head.fit(x[:160], y[:160])
    pred = head.predict(x[160:])
    assert pred.shape == (40, 3)


# ---------------------------------------------------------------------------
# XGBoostHead — predict_proba returns None for regression
# ---------------------------------------------------------------------------

import validation.heads.gp_matern  # noqa: F401
import validation.heads.gp_rbf  # noqa: F401
import validation.heads.gp_tanimoto  # noqa: F401


def test_gp_rbf_head_regression_fits_and_predicts(synthetic_regression_linear) -> None:
    x_train, y_train, x_test, y_test = synthetic_regression_linear
    cfg = OmegaConf.create(
        {
            "head_type": "gp-rbf",
            "objective_type": "regression",
            "num_inducing_cap": 64,
            "n_iter": 50,
            "lr": 1.0e-2,
            "batch_size": 64,
            "seed": 0,
        },
    )
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train.ravel())
    pred = head.predict(x_test)
    assert pred.shape == (x_test.shape[0],)
    r2 = 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    assert r2 > 0.3


def test_gp_rbf_inducing_count_capped_by_n_train_over_2() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8)).astype(np.float32)
    y = rng.standard_normal((200,)).astype(np.float32)
    cfg = OmegaConf.create(
        {
            "head_type": "gp-rbf",
            "objective_type": "regression",
            "num_inducing_cap": 1024,
            "n_iter": 5,
            "lr": 1.0e-2,
            "batch_size": 64,
            "seed": 0,
        },
    )
    head = load_head_from_config(cfg)
    head.fit(x[:160], y[:160])
    # 160 train rows → M = min(1024, 80) = 80
    assert head._gp.variational_strategy.inducing_points.shape[0] == 80  # noqa: SLF001


def test_gp_matern_head_regression_fits_and_predicts(synthetic_regression_linear) -> None:
    x_train, y_train, x_test, y_test = synthetic_regression_linear
    cfg = OmegaConf.create(
        {
            "head_type": "gp-matern",
            "objective_type": "regression",
            "num_inducing_cap": 64,
            "n_iter": 50,
            "lr": 1.0e-2,
            "batch_size": 64,
            "nu": 2.5,
            "seed": 0,
        },
    )
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train.ravel())
    pred = head.predict(x_test)
    r2 = 1 - ((y_test - pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    assert r2 > 0.3


def _binary_fingerprint_dataset(rng: np.random.Generator, n: int = 200, dim: int = 64):
    x = (rng.random((n, dim)) > 0.6).astype(np.float32)
    w = rng.standard_normal(dim).astype(np.float32)
    y = (x @ w + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return x, y


def test_gp_tanimoto_head_accepts_binary_input() -> None:
    rng = np.random.default_rng(11)
    x, y = _binary_fingerprint_dataset(rng)
    cfg = OmegaConf.create(
        {
            "head_type": "gp-tanimoto",
            "objective_type": "regression",
            "num_inducing_cap": 32,
            "n_iter": 30,
            "lr": 5.0e-2,
            "batch_size": 64,
            "seed": 0,
        },
    )
    head = load_head_from_config(cfg)
    head.fit(x[:160], y[:160])
    pred = head.predict(x[160:])
    assert pred.shape[0] == 40


def test_gp_tanimoto_head_rejects_continuous_input() -> None:
    rng = np.random.default_rng(12)
    x = rng.standard_normal((200, 64)).astype(np.float32)
    y = rng.standard_normal(200).astype(np.float32)
    cfg = OmegaConf.create(
        {
            "head_type": "gp-tanimoto",
            "objective_type": "regression",
            "num_inducing_cap": 32,
            "n_iter": 5,
            "lr": 5.0e-2,
            "batch_size": 64,
            "seed": 0,
        },
    )
    head = load_head_from_config(cfg)
    with pytest.raises(NotImplementedError, match="binary"):
        head.fit(x, y)


# ---------------------------------------------------------------------------
# XGBoostHead — predict_proba returns None for regression
# ---------------------------------------------------------------------------

def test_xgboost_head_predict_proba_none_for_regression(synthetic_regression) -> None:
    x_train, y_train, x_test, _ = synthetic_regression
    cfg = OmegaConf.create({
        "head_type": "xgboost",
        "objective_type": "regression",
        "path": "/tmp/xgb_test_reg2.json",
        "num_boost_round": 10,
        "params": {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.3, "device": "cpu", "seed": 42},
    })
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train)
    assert head.predict_proba(x_test) is None


# ---------------------------------------------------------------------------
# TabPFNHead (HPC-only — requires checkpoint download)
# ---------------------------------------------------------------------------

REQUIRES_TABPFN = pytest.mark.skipif(
    "TABPFN_AVAILABLE" not in os.environ or "TABPFN_TOKEN" not in os.environ,
    reason="TabPFN requires license token + checkpoint; set TABPFN_AVAILABLE=1 and TABPFN_TOKEN=<token>",
)


@REQUIRES_TABPFN
def test_tabpfn_head_regression_fits_and_predicts(synthetic_regression) -> None:
    import validation.heads.tabpfn  # noqa: F401
    x_train, y_train, x_test, _ = synthetic_regression
    cfg = OmegaConf.create(
        {
            "head_type": "tabpfn",
            "objective_type": "regression",
            "device": "cpu",
            "seed": 42,
        },
    )
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train.ravel())
    pred = head.predict(x_test).reshape(-1, 1)
    assert pred.shape == (x_test.shape[0], 1)


@REQUIRES_TABPFN
def test_tabpfn_head_per_target_wrapping_for_multioutput() -> None:
    import validation.heads.tabpfn  # noqa: F401
    rng = np.random.default_rng(13)
    x = rng.standard_normal((100, 8)).astype(np.float32)
    w = rng.standard_normal((8, 3)).astype(np.float32)
    y = (x @ w + 0.1 * rng.standard_normal((100, 3))).astype(np.float32)
    cfg = OmegaConf.create(
        {
            "head_type": "tabpfn",
            "objective_type": "regression",
            "device": "cpu",
            "seed": 42,
        },
    )
    head = load_head_from_config(cfg)
    head.fit(x[:80], y[:80])
    pred = head.predict(x[80:])
    assert pred.shape == (20, 3)


# ---------------------------------------------------------------------------
# TabICLHead (HPC-only — requires checkpoint download)
# ---------------------------------------------------------------------------

REQUIRES_TABICL = pytest.mark.skipif(
    "TABICL_AVAILABLE" not in os.environ,
    reason="TabICL requires checkpoint download; set TABICL_AVAILABLE=1 to enable",
)


@REQUIRES_TABICL
def test_tabicl_head_regression_fits_and_predicts(synthetic_regression) -> None:
    import validation.heads.tabicl  # noqa: F401
    x_train, y_train, x_test, _ = synthetic_regression
    cfg = OmegaConf.create(
        {
            "head_type": "tabicl",
            "objective_type": "regression",
            "device": "cpu",
            "seed": 42,
        },
    )
    head = load_head_from_config(cfg)
    head.fit(x_train, y_train.ravel())
    pred = head.predict(x_test).reshape(-1, 1)
    assert pred.shape == (x_test.shape[0], 1)


# ---------------------------------------------------------------------------
# GP heads on WIDE standardized features (2026-09-29). The tests above use 5 columns, where the default
# lengthscale (0.693) still sees neighbours; at embedding width the kernel was exp(-1500) = 0, its
# gradient too, and every prediction the same constant (ROC-AUC exactly 0.500 on the grid).
# ---------------------------------------------------------------------------


def _wide_embedding(rng: np.random.Generator, n: int = 300, dim: int = 768, rank: int = 8):
    """Low intrinsic dimension embedded in `dim` columns, like a model embedding (isotropic noise in all
    768 columns would defeat any isotropic kernel at n=240, fixed or not). StandardScaled, as the grid does."""
    z = rng.standard_normal((n, rank))
    x = z @ rng.standard_normal((rank, dim)) + 0.3 * rng.standard_normal((n, dim))
    x = (x - x.mean(0)) / x.std(0)
    return x.astype(np.float32), z


def _wide_classification(rng: np.random.Generator, n: int = 300, dim: int = 768):
    x, z = _wide_embedding(rng, n, dim)
    labels = (z[:, 0] + z[:, 1] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    return x, labels


def _gp_cfg(head_type: str, objective: str, **extra) -> OmegaConf:
    return OmegaConf.create({"head_type": head_type, "objective_type": objective, "num_inducing_cap": 128,
                             "n_iter": 30, "lr": 1.0e-2, "batch_size": 256, "seed": 0, **extra})


@pytest.mark.parametrize("head_type", ["gp-rbf", "gp-matern"])
def test_gp_classification_fits_on_wide_features(head_type: str) -> None:
    from sklearn.metrics import roc_auc_score
    x, y = _wide_classification(np.random.default_rng(3))
    head = load_head_from_config(_gp_cfg(head_type, "classification"))
    head.fit(x[:240], y[:240])
    p = head.predict_proba(x[240:])[:, 1]
    assert p.std() > 1e-3, "constant predictions: the kernel never saw a neighbour"
    assert roc_auc_score(y[240:], p) > 0.6


def test_gp_legacy_init_reproduces_the_constant_prediction() -> None:
    """lengthscale_init='default' keeps the published estimator reachable -- and shows the defect."""
    x, y = _wide_classification(np.random.default_rng(3))
    head = load_head_from_config(_gp_cfg("gp-rbf", "classification", lengthscale_init="default"))
    head.fit(x[:240], y[:240])
    assert head.predict_proba(x[240:])[:, 1].std() < 1e-4


@pytest.mark.parametrize("head_type", ["gp-rbf", "gp-matern"])
def test_gp_regression_offset_target_on_wide_features(head_type: str) -> None:
    rng = np.random.default_rng(5)
    x, z = _wide_embedding(rng)
    y = (-3.0 + 0.8 * z[:, 0] - 0.5 * z[:, 1] + 0.1 * rng.standard_normal(300)).astype(np.float32)
    head = load_head_from_config(_gp_cfg(head_type, "regression"))
    head.fit(x[:240], y[:240])
    pred = head.predict(x[240:])
    yt = y[240:]
    r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    assert abs(pred.mean() - yt.mean()) < 0.5, "target mean not restored"
    assert r2 > 0.2

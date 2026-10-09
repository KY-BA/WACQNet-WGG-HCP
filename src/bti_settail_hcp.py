"""Self-supervised set representation for background-tail transport.

BTI-SetTail-HCP treats every 32x32 OCO-3 background as one unordered set of
pixel observations augmented by fixed spatial coordinates.  The encoder is
trained without emission truth, HPR residuals, APE, background labels, or
background type.  A separate low-capacity ridge head maps the frozen scene
embedding to a background-level A4 score-tail target.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import random

import numpy as np
import pandas as pd


PIXEL_CHANNELS = (
    "background_residual",
    "background_low_centered",
    "log_neighbor_count",
    "residual_gradient_x",
    "residual_gradient_y",
    "x_coordinate",
    "y_coordinate",
    "valid_indicator",
)


def deployable_scale_elementwise(frame: pd.DataFrame, fit, correction_clip: float) -> np.ndarray:
    """Numerically equivalent A4 scale without a large BLAS matrix product.

    The legacy Python-3.8 runtime used for TensorFlow can deadlock in its BLAS
    implementation on the 243000x5 product.  Summing the same five terms
    elementwise preserves the frozen formula and avoids that runtime defect.
    """
    raw = np.zeros(len(frame), dtype=float)
    for name, center, scale, coefficient in zip(
        fit.feature_names, fit.feature_center, fit.feature_scale, fit.coefficients
    ):
        raw += ((frame[name].to_numpy(float) - float(center)) / float(scale)) * float(coefficient)
    correction = np.clip(raw - float(fit.correction_center), -float(correction_clip), float(correction_clip))
    result = frame.base_scale.to_numpy(float) * np.exp(correction)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise RuntimeError("Invalid elementwise A4 scale")
    return result


def deterministic_folds(groups, n_folds: int, seed: int) -> dict[str, int]:
    unique = sorted(set(map(str, groups)))
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest(),
    )
    return {group: index % int(n_folds) for index, group in enumerate(ranked)}


def finite_order_quantile(values, probability: float) -> float:
    ordered = np.sort(np.asarray(values, float))
    if len(ordered) == 0:
        raise ValueError("Cannot compute an order statistic from no values")
    rank = min(len(ordered), max(1, int(np.ceil((len(ordered) + 1) * probability))))
    return float(ordered[rank - 1])


def rank_spearman(values_a, values_b) -> float:
    """Average-rank Spearman correlation without BLAS-backed corrcoef."""
    a = pd.Series(np.asarray(values_a, float)).rank(method="average").to_numpy(float)
    b = pd.Series(np.asarray(values_b, float)).rank(method="average").to_numpy(float)
    ac = a - float(np.mean(a)); bc = b - float(np.mean(b))
    denominator = float(np.sqrt(np.sum(ac * ac) * np.sum(bc * bc)))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(ac * bc) / denominator)


def load_pixel_scene(path) -> tuple[np.ndarray, np.ndarray]:
    """Return a 1024x8 deployable pixel set and a 192-d clean target."""
    with np.load(path) as scene:
        residual = np.asarray(scene["background_residual"], float)
        low = np.asarray(scene["background_low"], float)
        neighbor = np.asarray(scene["neighbor_count"], float)
        if "valid_mask" in scene.files:
            valid = np.asarray(scene["valid_mask"], bool)
        else:
            valid = np.isfinite(residual) & np.isfinite(low) & (neighbor > 0)
    if residual.shape != (32, 32) or low.shape != (32, 32) or valid.shape != (32, 32):
        raise ValueError(f"Expected frozen 32x32 scene: {path}")
    valid &= np.isfinite(residual) & np.isfinite(low) & np.isfinite(neighbor)
    if valid.sum() < 32:
        raise ValueError(f"Insufficient valid observations: {path}")
    r = np.where(valid, residual, 0.0)
    low_center = np.where(valid, low - np.median(low[valid]), 0.0)
    support = np.where(valid, np.log1p(np.maximum(neighbor, 0.0)), 0.0)
    gx = np.zeros_like(r); gy = np.zeros_like(r)
    gx[:, 1:-1] = 0.5 * (r[:, 2:] - r[:, :-2])
    gy[1:-1, :] = 0.5 * (r[2:, :] - r[:-2, :])
    gx *= valid; gy *= valid
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, 32), np.linspace(-1.0, 1.0, 32), indexing="ij")
    pixels = np.stack([r, low_center, support, gx, gy, xx, yy, valid.astype(float)], axis=-1)

    # A clean multiscale target: 8x8 block means for residual, low field, and mask.
    target_parts = []
    for array in (r, low_center, valid.astype(float)):
        target_parts.append(array.reshape(8, 4, 8, 4).mean(axis=(1, 3)).reshape(-1))
    target = np.concatenate(target_parts).astype(np.float32)
    return pixels.reshape(-1, len(PIXEL_CHANNELS)).astype(np.float32), target


@dataclass(frozen=True)
class PixelStandardizer:
    center: tuple[float, ...]
    scale: tuple[float, ...]


def fit_pixel_standardizer(pixel_scenes: np.ndarray) -> PixelStandardizer:
    x = np.asarray(pixel_scenes, float)
    valid = x[..., 7] > 0.5
    center = np.zeros(x.shape[-1], float)
    scale = np.ones(x.shape[-1], float)
    # Observation channels only; coordinates and valid indicator stay fixed.
    for channel in range(5):
        values = x[..., channel][valid]
        center[channel] = np.median(values)
        robust = 1.4826 * np.median(np.abs(values - center[channel]))
        scale[channel] = robust if robust > 1e-6 else max(np.std(values), 1.0)
    return PixelStandardizer(tuple(map(float, center)), tuple(map(float, scale)))


def transform_pixels(pixel_scenes: np.ndarray, fit: PixelStandardizer) -> np.ndarray:
    x = np.asarray(pixel_scenes, np.float32).copy()
    center = np.asarray(fit.center, np.float32)
    scale = np.asarray(fit.scale, np.float32)
    x[..., :5] = np.clip((x[..., :5] - center[:5]) / scale[:5], -6.0, 6.0)
    invalid = x[..., 7] <= 0.5
    for channel in range(5):
        x[..., channel][invalid] = 0.0
    return x


@dataclass(frozen=True)
class TargetStandardizer:
    center: tuple[float, ...]
    scale: tuple[float, ...]


def fit_target_standardizer(targets: np.ndarray) -> TargetStandardizer:
    center = np.mean(targets, axis=0)
    scale = np.std(targets, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return TargetStandardizer(tuple(map(float, center)), tuple(map(float, scale)))


def transform_targets(targets: np.ndarray, fit: TargetStandardizer) -> np.ndarray:
    return ((np.asarray(targets) - np.asarray(fit.center)) / np.asarray(fit.scale)).astype(np.float32)


def augment_pixel_sets(x: np.ndarray, *, copies: int, mask_fraction: float,
                       noise_sd: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = []
    for scene in x:
        valid_positions = np.flatnonzero(scene[:, 7] > 0.5)
        for _ in range(int(copies)):
            view = scene.copy()
            count = int(np.floor(mask_fraction * len(valid_positions)))
            if count:
                removed = rng.choice(valid_positions, count, replace=False)
                view[removed, :5] = 0.0
                view[removed, 7] = 0.0
            keep = view[:, 7] > 0.5
            view[keep, :5] += rng.normal(0.0, noise_sd, size=(keep.sum(), 5)).astype(np.float32)
            result.append(view)
    return np.asarray(result, np.float32)


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(seed); np.random.seed(seed)


def build_set_autoencoder(*, pixel_features: int, target_features: int,
                          latent_dim: int, seed: int):
    _set_determinism(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(1024, pixel_features), name="pixel_set")
    phi = tf.keras.layers.Dense(16, activation="relu", name="phi_1")(inputs)
    phi = tf.keras.layers.Dense(8, activation="relu", name="phi_2")(phi)
    valid = inputs[..., 7:8]
    denominator = tf.keras.layers.Lambda(
        lambda value: tf.maximum(tf.reduce_sum(value, axis=1), 1.0), name="valid_count"
    )(valid)
    mean = tf.keras.layers.Lambda(
        lambda args: tf.reduce_sum(args[0] * args[1], axis=1) / args[2], name="set_mean"
    )([phi, valid, denominator])
    second = tf.keras.layers.Lambda(
        lambda args: tf.reduce_sum(tf.square(args[0]) * args[1], axis=1) / args[2],
        name="set_second_moment",
    )([phi, valid, denominator])
    std = tf.keras.layers.Lambda(
        lambda args: tf.sqrt(tf.maximum(args[0] - tf.square(args[1]), 1e-6)), name="set_std"
    )([second, mean])
    pooled = tf.keras.layers.Concatenate(name="mean_std_pool")([mean, std])
    hidden = tf.keras.layers.Dense(8, activation="tanh", name="rho")(pooled)
    latent = tf.keras.layers.Dense(latent_dim, activation="linear", name="scene_embedding")(hidden)
    decoded = tf.keras.layers.Dense(32, activation="relu", name="decoder_hidden")(latent)
    outputs = tf.keras.layers.Dense(target_features, activation="linear", name="clean_signature")(decoded)
    model = tf.keras.Model(inputs, outputs, name="bti_settail_autoencoder")
    encoder = tf.keras.Model(inputs, latent, name="bti_settail_encoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model, encoder


def train_set_encoder(pixel_scenes: np.ndarray, targets: np.ndarray, *, cfg: dict,
                      seed: int):
    pixel_fit = fit_pixel_standardizer(pixel_scenes)
    target_fit = fit_target_standardizer(targets)
    clean = transform_pixels(pixel_scenes, pixel_fit)
    augmented = augment_pixel_sets(
        clean,
        copies=int(cfg["augmentation_copies"]),
        mask_fraction=float(cfg["mask_fraction"]),
        noise_sd=float(cfg["noise_sd"]),
        seed=seed,
    )
    target_standard = transform_targets(targets, target_fit)
    repeated_targets = np.repeat(target_standard, int(cfg["augmentation_copies"]), axis=0)
    model, encoder = build_set_autoencoder(
        pixel_features=clean.shape[-1], target_features=targets.shape[-1],
        latent_dim=int(cfg["latent_dim"]), seed=seed,
    )
    history = model.fit(
        augmented, repeated_targets,
        epochs=int(cfg["epochs"]), batch_size=int(cfg["batch_size"]),
        shuffle=False, verbose=0,
    )
    return encoder, pixel_fit, target_fit, history.history


def encode_pixel_scenes(encoder, pixel_scenes: np.ndarray, fit: PixelStandardizer) -> np.ndarray:
    clean = transform_pixels(pixel_scenes, fit)
    return np.asarray(encoder.predict(clean, verbose=0), float)


@dataclass(frozen=True)
class TailRidgeFit:
    coefficients: tuple[float, ...]
    intercept: float
    prediction_center: float
    ridge_alpha: float
    shrinkage: float
    log_clip: float


def solve_small_system(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Partial-pivot Gaussian elimination for the fixed three-dimensional head."""
    a = np.asarray(matrix, float).copy()
    b = np.asarray(vector, float).copy()
    n = len(b)
    if a.shape != (n, n) or n > 8:
        raise ValueError("Only small square systems are supported")
    for column in range(n):
        pivot = column + int(np.argmax(np.abs(a[column:, column])))
        if abs(a[pivot, column]) < 1e-12:
            raise RuntimeError("Singular ridge system")
        if pivot != column:
            a[[column, pivot]] = a[[pivot, column]]
            b[[column, pivot]] = b[[pivot, column]]
        for row in range(column + 1, n):
            factor = a[row, column] / a[column, column]
            a[row, column:] -= factor * a[column, column:]
            b[row] -= factor * b[column]
    result = np.zeros(n, float)
    for row in range(n - 1, -1, -1):
        remainder = float(np.sum(a[row, row + 1:] * result[row + 1:]))
        result[row] = (b[row] - remainder) / a[row, row]
    return result


def fit_tail_ridge(embedding: np.ndarray, log_tail: np.ndarray, *, ridge_alpha: float,
                   shrinkage: float, log_clip: float) -> TailRidgeFit:
    x = np.asarray(embedding, float); y = np.asarray(log_tail, float)
    xc = x - x.mean(axis=0); yc = y - y.mean()
    dimension = x.shape[1]
    gram = np.empty((dimension, dimension), float)
    for row in range(dimension):
        for column in range(dimension):
            gram[row, column] = float(np.sum(xc[:, row] * xc[:, column]))
    gram += ridge_alpha * np.eye(dimension)
    rhs = np.asarray([float(np.sum(xc[:, column] * yc)) for column in range(dimension)])
    beta = solve_small_system(gram, rhs)
    intercept = float(y.mean() - np.sum(x.mean(axis=0) * beta))
    prediction = intercept + np.sum(x * beta[None, :], axis=1)
    return TailRidgeFit(tuple(map(float, beta)), intercept, float(prediction.mean()),
                        float(ridge_alpha), float(shrinkage), float(log_clip))


def tail_factor(embedding: np.ndarray, fit: TailRidgeFit) -> np.ndarray:
    x = np.asarray(embedding, float)
    prediction = fit.intercept + np.sum(x * np.asarray(fit.coefficients)[None, :], axis=1)
    correction = fit.shrinkage * (prediction - fit.prediction_center)
    return np.exp(np.clip(correction, -fit.log_clip, fit.log_clip))


def group_tail_targets(frame: pd.DataFrame, scale: np.ndarray, probability: float) -> pd.DataFrame:
    scores = frame.absolute_error.to_numpy(float) / np.asarray(scale, float)
    rows = []
    group_values = frame.background_id.astype(str).to_numpy()
    for group in sorted(set(group_values)):
        values = scores[group_values == group]
        tail = finite_order_quantile(values, probability)
        rows.append({"background_id": group, "tail_score": tail,
                     "log_tail_score": float(np.log(tail + 1e-8)), "n": len(values)})
    return pd.DataFrame(rows)


def cross_fitted_tail_factors(frame: pd.DataFrame, embeddings: pd.DataFrame,
                              a4_scale: np.ndarray, *, cfg: dict):
    targets = group_tail_targets(frame, a4_scale, float(cfg["group_tail_probability"]))
    joined = embeddings.merge(targets, on="background_id", validate="one_to_one")
    latent_columns = [column for column in joined if column.startswith("z")]
    folds = deterministic_folds(joined.background_id, int(cfg["crossfit_folds"]),
                                int(cfg["crossfit_seed"]))
    group_factor = {}; audit = []
    for fold in range(int(cfg["crossfit_folds"])):
        validation = {group for group, value in folds.items() if value == fold}
        train = joined.loc[~joined.background_id.isin(validation)]
        held = joined.loc[joined.background_id.isin(validation)]
        fit = fit_tail_ridge(
            train[latent_columns].to_numpy(), train.log_tail_score.to_numpy(),
            ridge_alpha=float(cfg["ridge_alpha"]), shrinkage=float(cfg["shrinkage"]),
            log_clip=float(cfg["log_correction_clip"]),
        )
        factors = tail_factor(held[latent_columns].to_numpy(), fit)
        for group, factor in zip(held.background_id.astype(str), factors):
            group_factor[group] = float(factor)
            audit.append({"background_id": group, "inner_fold": fold, "factor": float(factor)})
    mapped = frame.background_id.astype(str).map(group_factor).to_numpy(float)
    if not np.isfinite(mapped).all() or (mapped <= 0).any():
        raise RuntimeError("Incomplete group-cross-fitted SetTail factors")
    final_fit = fit_tail_ridge(
        joined[latent_columns].to_numpy(), joined.log_tail_score.to_numpy(),
        ridge_alpha=float(cfg["ridge_alpha"]), shrinkage=float(cfg["shrinkage"]),
        log_clip=float(cfg["log_correction_clip"]),
    )
    return mapped, final_fit, targets, pd.DataFrame(audit)

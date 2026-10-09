"""BTI-WACQNet model and deterministic deployable feature construction."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import tensorflow as tf


def stable_seed(base: int, *parts: object) -> int:
    token = "|".join([str(base), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def _block_mean_2x(array: np.ndarray) -> np.ndarray:
    if array.shape != (32, 32):
        raise ValueError(f"Expected 32x32 array, got {array.shape}")
    return array.reshape(16, 2, 16, 2).mean(axis=(1, 3))


def build_spatial_features(
    image: np.ndarray,
    valid: np.ndarray,
    template: np.ndarray,
    source_yx: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Create the frozen six spatial and three scalar inputs.

    The first scalar (log1p HPR anchor) is inserted by the caller.
    """
    xco2 = np.asarray(image[..., 0], np.float32)
    u = np.asarray(image[..., 1], np.float32)
    v = np.asarray(image[..., 2], np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(xco2)
    if not valid.any():
        raise ValueError("No valid XCO2 pixel")
    values = xco2[valid]
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    robust_sigma = max(1.4826 * mad, 0.05)
    filled = np.where(valid, xco2, center)
    centered = np.clip((filled - center) / 6.0, -2.0, 2.0)
    scaled = np.clip((filled - center) / robust_sigma, -6.0, 6.0) / 3.0

    uu = float(np.nanmedian(u))
    vv = float(np.nanmedian(v))
    speed = float(np.hypot(uu, vv))
    norm = max(speed, 1.0e-6)
    ux, uy = uu / norm, -vv / norm  # image x/right, y/down
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    dx, dy = xx - float(source_yx[1]), yy - float(source_yx[0])
    downwind = np.clip((dx * ux + dy * uy) / 16.0, -1.5, 1.5)
    crosswind = np.clip((-dx * uy + dy * ux) / 16.0, -1.5, 1.5)
    tmpl = np.asarray(template, np.float32)
    tmpl = tmpl / max(float(np.nanmax(tmpl)), 1.0e-6)
    tmpl = np.where(valid, tmpl, 0.0)

    channels = [centered, scaled, valid.astype(np.float32), tmpl, downwind, crosswind]
    spatial = np.stack([_block_mean_2x(ch) for ch in channels], axis=-1).astype(np.float32)
    observed = np.asarray([np.log1p(max(speed, 0.0)), float(valid.mean())], np.float32)
    return spatial, observed


def pinball(residual: tf.Tensor, tau: float) -> tf.Tensor:
    tau_t = tf.cast(tau, residual.dtype)
    return tf.maximum(tau_t * residual, (tau_t - 1.0) * residual)


def sample_loss(
    y_true: tf.Tensor,
    outputs: dict[str, tf.Tensor],
    *,
    point_delta: float = 0.30,
) -> tf.Tensor:
    y = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    mu = tf.reshape(outputs["mu"], [-1])
    s_minus = tf.reshape(outputs["s_minus"], [-1])
    s_plus = tf.reshape(outputs["s_plus"], [-1])
    delta = tf.reshape(outputs["delta"], [-1])
    denom = y + 1.0
    rel = (mu - y) / denom
    abs_rel = tf.abs(rel)
    huber = tf.where(abs_rel <= point_delta, 0.5 * tf.square(rel), point_delta * (abs_rel - 0.5 * point_delta))
    lower_residual = (y - (mu - s_minus)) / denom
    upper_residual = (y - (mu + s_plus)) / denom
    qloss = 0.5 * pinball(lower_residual, 0.05) + 0.5 * pinball(upper_residual, 0.95)
    width = 0.01 * (s_minus + s_plus) / denom
    anchor = 0.02 * tf.square(delta)
    return huber + qloss + width + anchor


def _masked_pool(inputs: list[tf.Tensor]) -> tf.Tensor:
    feature, weight = inputs
    weight = tf.maximum(weight, 0.0)
    numerator = tf.reduce_sum(feature * weight, axis=(1, 2))
    denominator = tf.reduce_sum(weight, axis=(1, 2)) + 1.0e-6
    return numerator / denominator


def build_wacqnet(config: dict[str, Any], ablation: dict[str, Any]) -> tf.keras.Model:
    arch = config["architecture"]
    spatial = tf.keras.Input(shape=(16, 16, 6), name="spatial")
    scalar = tf.keras.Input(shape=(3,), name="scalar")

    use_wind = bool(ablation["use_wind_coordinates"])
    if use_wind:
        selected = spatial
    else:
        selected = tf.keras.layers.Lambda(lambda value: value[..., :3], name="remove_wind_channels")(spatial)

    x = selected
    for index, filters in enumerate(arch["conv_filters"]):
        stride = 1 if index == 0 else 2
        residual = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same", name=f"block{index}_skip")(x)
        y = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", name=f"block{index}_conv1")(x)
        y = tf.keras.layers.LayerNormalization(axis=-1, name=f"block{index}_ln1")(y)
        y = tf.keras.layers.Activation("swish", name=f"block{index}_act1")(y)
        y = tf.keras.layers.Conv2D(filters, 3, padding="same", name=f"block{index}_conv2")(y)
        y = tf.keras.layers.LayerNormalization(axis=-1, name=f"block{index}_ln2")(y)
        x = tf.keras.layers.Activation("swish", name=f"block{index}_out")(residual + y)

    global_mean = tf.keras.layers.GlobalAveragePooling2D(name="global_mean")(x)
    global_max = tf.keras.layers.GlobalMaxPooling2D(name="global_max")(x)
    pieces = [global_mean, global_max]
    if bool(ablation["use_contrast"]):
        valid = tf.keras.layers.Lambda(lambda value: value[..., 2:3], name="valid_channel")(spatial)
        template = tf.keras.layers.Lambda(lambda value: value[..., 3:4], name="template_channel")(spatial)
        valid4 = tf.keras.layers.AveragePooling2D(pool_size=4, name="valid_pool4")(valid)
        tmpl4 = tf.keras.layers.AveragePooling2D(pool_size=4, name="template_pool4")(template)
        fg_weight = tf.keras.layers.Multiply(name="fg_weight")([valid4, tmpl4])
        bg_weight = tf.keras.layers.Multiply(name="bg_weight")([valid4, 1.0 - tmpl4])
        fg = tf.keras.layers.Lambda(_masked_pool, name="plume_pool")([x, fg_weight])
        bg = tf.keras.layers.Lambda(_masked_pool, name="background_pool")([x, bg_weight])
        contrast = tf.keras.layers.Subtract(name="plume_minus_background")([fg, bg])
        pieces += [fg, bg, contrast]

    c = scalar
    for index, units in enumerate(arch["condition_mlp_units"]):
        c = tf.keras.layers.Dense(units, activation="swish", name=f"condition_dense{index}")(c)
    pieces.append(c)
    fused = tf.keras.layers.Concatenate(name="fusion_concat")(pieces)
    for index, units in enumerate(arch["fusion_units"]):
        fused = tf.keras.layers.Dense(units, activation="swish", name=f"fusion_dense{index}")(fused)
        fused = tf.keras.layers.Dropout(float(arch["dropout"]), name=f"fusion_dropout{index}")(fused)

    max_corr = float(arch["max_log_correction"])
    delta = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer="zeros", bias_initializer="zeros", name="delta_raw")(fused)
    delta = tf.keras.layers.Lambda(lambda value: max_corr * value, name="delta")(delta)
    anchor_log = tf.keras.layers.Lambda(lambda value: value[:, :1], name="anchor_log")(scalar)
    corrected_log = tf.keras.layers.ReLU(name="nonnegative_log_emission")(anchor_log + delta)
    mu = tf.keras.layers.Lambda(lambda value: tf.math.expm1(value), name="mu")(corrected_log)
    s_minus = tf.keras.layers.Dense(1, activation="softplus", bias_initializer=tf.keras.initializers.Constant(1.0), name="s_minus")(fused)
    s_plus = tf.keras.layers.Dense(1, activation="softplus", bias_initializer=tf.keras.initializers.Constant(1.0), name="s_plus")(fused)
    risk = tf.keras.layers.Lambda(
        lambda values: (values[0] + values[1]) / (2.0 * (values[2] + 1.0)), name="risk"
    )([s_minus, s_plus, mu])
    return tf.keras.Model([spatial, scalar], {"mu": mu, "s_minus": s_minus, "s_plus": s_plus, "risk": risk, "delta": delta}, name=ablation["id"])


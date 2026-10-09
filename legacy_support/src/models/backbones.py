"""Adapters around the existing TensorFlow/Keras HPR and CNN assets."""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import tensorflow as tf

from src.utils.checkpoint import load_keras_weights


class BackboneAdapter(tf.keras.Model):
    """Uniform feature interface over an existing Keras feature extractor."""

    def __init__(self, backbone: tf.keras.Model, hpr_feature_layer: str | None = None, name: str = "backbone_adapter"):
        super().__init__(name=name)
        self.backbone = backbone
        self.hpr_feature_layer = hpr_feature_layer
        self._feature_model: tf.keras.Model | None = None
        if hpr_feature_layer is not None:
            try:
                layer = backbone.get_layer(hpr_feature_layer)
            except ValueError as exc:
                raise ValueError(f"HPR feature layer not found: {hpr_feature_layer}") from exc
            self._feature_model = tf.keras.Model(backbone.input, [layer.output, backbone.output])

    def call(self, image: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor | None]:
        # A frozen pretrained backbone must stay in inference mode in Stage 2.
        # This preserves legacy BatchNorm moving statistics and avoids asking
        # TensorFlow 2.9 for a deterministic frozen-BN backward kernel.
        backbone_training = bool(training and self.backbone.trainable)
        if self._feature_model is None:
            feature = self.backbone(image, training=backbone_training)
            hpr = feature if "hierarchical" in self.backbone.name else None
        else:
            hpr, feature = self._feature_model(image, training=backbone_training)
        if not self.backbone.trainable:
            feature = tf.stop_gradient(feature)
            if hpr is not None:
                hpr = tf.stop_gradient(hpr)
        if feature.shape.rank == 4:
            feature_map = feature
            global_feature = tf.reduce_mean(feature, axis=(1, 2))
        else:
            feature_map = None
            global_feature = tf.reshape(feature, [tf.shape(feature)[0], -1])
        return {"feature_map": feature_map, "global_feature": global_feature, "hpr_feature": hpr}

    def freeze_all(self) -> None:
        self.backbone.trainable = False

    def unfreeze_last_blocks(self, patterns: Iterable[str], count: int = 1) -> list[str]:
        """Unfreeze the final matching named blocks while all other layers stay frozen."""
        self.backbone.trainable = True
        leaves = list(_leaf_layers(self.backbone))
        for layer in leaves:
            layer.trainable = False
        matches = [layer for layer in leaves if any(pattern in layer.name for pattern in patterns)]
        if not matches:
            raise ValueError(f"No backbone layer matched patterns: {list(patterns)}")
        selected = matches[-count:]
        for layer in selected:
            layer.trainable = True
        return [layer.name for layer in selected]


def _leaf_layers(model: tf.keras.Model) -> Iterable[tf.keras.layers.Layer]:
    """Yield nested leaf layers while leaving Model containers trainable."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True
            yield from _leaf_layers(layer)
        else:
            yield layer


@dataclass
class LegacyLoadResult:
    full_model: tf.keras.Model
    backbone: BackboneAdapter
    report: dict[str, Any]


def build_legacy_model(
    legacy_repo: str | Path,
    model_name: str,
    input_shape: tuple[int, int, int],
    checkpoint: str | Path | None = None,
    dropout_rate: float = 0.2,
    report_path: str | Path | None = None,
    normalization_layer: str | Path | None = None,
) -> LegacyLoadResult:
    """Build the original project model, load its real checkpoint, and extract core features."""
    root = Path(legacy_repo).resolve()
    if not (root / "models" / "reg.py").exists():
        raise FileNotFoundError(f"Legacy Keras model factory not found under: {root}")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("models.reg")
        builder = module.Reg_model_builder(
            name=model_name,
            input_shape=list(input_shape),
            classes=1,
            dropout_rate=dropout_rate,
        )
        full_model = builder.get_model()
        # Build variables before loading and before Stage-1 inference.
        full_model(tf.zeros([1, *input_shape], dtype=tf.float32), training=False)
        report: dict[str, Any] = {"legacy_repo": str(root), "model_name": model_name}
        if checkpoint:
            # Exact legacy architecture: topology loading also covers unnamed
            # Dense layers that cannot be audited reliably with by-name HDF5.
            report.update(load_keras_weights(full_model, checkpoint, by_name=False, skip_mismatch=False, report_path=report_path))
        else:
            report["status"] = "random_initialization_not_valid_for_stage1"
        core = builder.core_model
        if normalization_layer:
            norm_path = Path(normalization_layer)
            if not norm_path.exists():
                raise FileNotFoundError(f"Legacy normalization layer missing: {norm_path}")
            preprocessing = importlib.import_module("models.preprocessing")
            custom_objects = {
                "TrainingTimeNormalization": preprocessing.TrainingTimeNormalization,
                "CloudsLayer": preprocessing.CloudsLayer,
                "ConditionalNoiseLayer": preprocessing.ConditionalNoiseLayer,
            }
            norm = tf.keras.models.load_model(str(norm_path), compile=False, custom_objects=custom_objects)
            inputs = tf.keras.Input(shape=input_shape, name="legacy_normalized_input")
            # The legacy custom layer intentionally normalizes only when
            # training=True; its historical data pipeline used the same call.
            normalized = norm(inputs, training=True)
            core = tf.keras.Model(inputs, builder.core_model(normalized), name=f"{builder.core_model.name}_with_legacy_norm")
            full_model = tf.keras.Model(inputs, full_model(normalized), name=f"{full_model.name}_with_legacy_norm")
            report["normalization_layer"] = str(norm_path.resolve())
        else:
            report["normalization_layer"] = None
            report["normalization_warning"] = "Input normalization was not supplied; Stage-1 equivalence must be verified before training."
        return LegacyLoadResult(full_model, BackboneAdapter(core), report)
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)

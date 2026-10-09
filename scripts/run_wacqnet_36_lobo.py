"""Run preregistered Development-only 36-background WACQNet LOBO."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wacqnet import build_wacqnet, sample_loss, stable_seed


for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass


def finite_quantile(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, float))
    rank = min(len(ordered), max(1, int(math.ceil((len(ordered) + 1) * probability))))
    return float(ordered[rank - 1])


def read_indices(h5: h5py.File, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(indices, int))
    return (
        np.asarray(h5["spatial"][ordered], np.float32),
        np.asarray(h5["scalar"][ordered], np.float32),
        np.asarray(h5["true_emission"][ordered], np.float32),
    )


def predict(model: tf.keras.Model, spatial: np.ndarray, scalar: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    values = {key: [] for key in ["mu", "s_minus", "s_plus", "risk", "delta"]}
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 16, 16, 6], tf.float32),
            tf.TensorSpec([None, 3], tf.float32),
        ],
        reduce_retracing=True,
    )
    def infer(x_spatial, x_scalar):
        out = model([x_spatial, x_scalar], training=False)
        return out["mu"], out["s_minus"], out["s_plus"], out["risk"], out["delta"]

    inference_batch = max(1024, batch_size)
    keys = list(values)
    for start in range(0, len(spatial), inference_batch):
        parts = infer(spatial[start:start + inference_batch], scalar[start:start + inference_batch])
        for key, part in zip(keys, parts):
            values[key].append(np.asarray(part).reshape(-1))
    return {key: np.concatenate(parts) for key, parts in values.items()}


def fit_model(
    cfg: dict,
    ablation: dict,
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    train_groups: np.ndarray,
    seed: int,
) -> tuple[tf.keras.Model, list[dict]]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    spatial, scalar, target = train_data
    model = build_wacqnet(cfg, ablation)
    optimizer = tf.keras.optimizers.Adam(float(cfg["training"]["learning_rate"]), clipnorm=float(cfg["training"]["gradient_clip_norm"]))
    batch_size = int(cfg["training"]["batch_size"])
    unique_groups = np.unique(train_groups)
    group_weights = {int(g): 1.0 for g in unique_groups}
    use_dro = bool(ablation["use_group_dro"])
    eta = float(cfg["loss"]["group_dro"]["eta"])
    clip_low, clip_high = map(float, cfg["loss"]["group_dro"]["group_weight_clip"])
    weight_decay = float(cfg["training"]["weight_decay"])
    lr = float(cfg["training"]["learning_rate"])

    @tf.function
    def train_step(x_spatial, x_scalar, y, weights):
        with tf.GradientTape() as tape:
            out = model([x_spatial, x_scalar], training=True)
            losses = sample_loss(y, out)
            objective = tf.reduce_sum(losses * weights) / (tf.reduce_sum(weights) + 1.0e-6)
        gradients = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for variable in model.trainable_variables:
            if "kernel" in variable.name:
                variable.assign_sub(lr * weight_decay * variable)
        return objective, losses

    rng = np.random.default_rng(seed)
    history: list[dict] = []
    for epoch in range(int(cfg["training"]["epochs"])):
        order = rng.permutation(len(target))
        sums = {int(g): 0.0 for g in unique_groups}
        counts = {int(g): 0 for g in unique_groups}
        objectives = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            weights = np.asarray([group_weights[int(g)] for g in train_groups[idx]], np.float32)
            objective, losses = train_step(spatial[idx], scalar[idx], target[idx], weights)
            objectives.append(float(objective.numpy()))
            loss_values = losses.numpy()
            for group in np.unique(train_groups[idx]):
                select = train_groups[idx] == group
                sums[int(group)] += float(loss_values[select].sum())
                counts[int(group)] += int(select.sum())
        group_loss = {g: sums[g] / max(counts[g], 1) for g in sums}
        if use_dro:
            raw = {g: group_weights[g] * math.exp(eta * group_loss[g]) for g in group_weights}
            mean_raw = float(np.mean(list(raw.values())))
            group_weights = {g: float(np.clip(raw[g] / mean_raw, clip_low, clip_high)) for g in raw}
            renorm = float(np.mean(list(group_weights.values())))
            group_weights = {g: value / renorm for g, value in group_weights.items()}
        history.append({
            "epoch": epoch + 1, "objective": float(np.mean(objectives)),
            "max_group_loss": float(max(group_loss.values())), "median_group_loss": float(np.median(list(group_loss.values()))),
            "max_group_weight": float(max(group_weights.values())), "min_group_weight": float(min(group_weights.values())),
        })
        print(
            f"{ablation['id']} epoch {epoch + 1}/{cfg['training']['epochs']} "
            f"objective={history[-1]['objective']:.6f} max_group={history[-1]['max_group_loss']:.6f}",
            flush=True,
        )
    return model, history


def metrics_for_background(frame: pd.DataFrame) -> dict:
    ape = np.abs(frame.true_emission - frame.mu) / frame.true_emission
    unreliable = (ape >= 0.30).astype(int)
    x = frame.true_emission.to_numpy(float)
    y = frame.mu.to_numpy(float)
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    slope = float(np.sum(xc * yc) / max(float(np.sum(xc * xc)), 1.0e-12))
    intercept = float(np.mean(y) - slope * np.mean(x))
    try:
        auroc = float(roc_auc_score(unreliable, frame.risk))
    except ValueError:
        auroc = float("nan")
    rank_a = pd.Series(frame.risk.to_numpy(float)).rank(method="average").to_numpy(float)
    rank_b = pd.Series(ape.to_numpy(float)).rank(method="average").to_numpy(float)
    rank_a -= float(np.mean(rank_a)); rank_b -= float(np.mean(rank_b))
    spearman = float(np.sum(rank_a * rank_b) / max(float(np.sqrt(np.sum(rank_a * rank_a) * np.sum(rank_b * rank_b))), 1.0e-12))
    return {
        "median_ape": float(np.median(ape)),
        "p_reliable30": float(np.mean(ape < 0.30)),
        "mae": float(np.mean(np.abs(frame.true_emission - frame.mu))),
        "scale_compression_slope": float(slope), "scale_compression_intercept": float(intercept),
        "spearman_risk_ape": spearman,
        "auroc_unreliable30": auroc,
        "auprc_unreliable30": float(average_precision_score(unreliable, frame.risk)),
        "unreliable_prevalence": float(unreliable.mean()),
        "picp90": float(frame.covered90.mean()),
        "mpiw": float(frame.width90.mean()),
        "normalized_width": float(np.mean(frame.width90 / (frame.mu + 1.0))),
        "lower_zero_fraction": float(np.mean(frame.lower90 <= 1.0e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="D:/CO2/_third/configs/wacqnet_development_protocol_v1.yaml")
    parser.add_argument("--candidates", nargs="+", default=["WACQ_FULL"])
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--fold-end", type=int, default=36)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    protocol = json.loads((Path(cfg["outputs"]["protocol_dir"]) / "protocol_freeze_audit.json").read_text(encoding="utf-8"))
    if protocol["status"] != "PASS" or protocol["protocol_version"] != cfg["protocol_version"]:
        raise RuntimeError("Stale protocol freeze")
    cache_dir = Path(cfg["outputs"]["cache_dir"])
    cache_path = cache_dir / "wacqnet_levelb_cache.h5"
    metadata = pd.read_csv(cache_dir / "wacqnet_levelb_metadata.csv")
    if metadata.background_id.nunique() != 36 or len(metadata) != 243000:
        raise RuntimeError("Cache metadata count drift")
    candidates = {x["id"]: x for x in cfg["ablations"]["trained"]}
    unknown = set(args.candidates) - set(candidates)
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")

    output = Path(cfg["outputs"]["training_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    metrics_path = output / f"wacqnet_lobo_background_metrics{suffix}.csv"
    predictions_path = output / f"wacqnet_lobo_predictions{suffix}.csv"
    history_path = output / f"wacqnet_training_history{suffix}.csv"
    progress_path = output / f"training_progress{suffix}.jsonl"
    completed = set()
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        completed = set(zip(old.candidate_id.astype(str), old.heldout_background.astype(str)))

    backgrounds = sorted(metadata.background_id.unique())
    train_reps = set(map(int, cfg["training"]["training_repetitions_per_condition"]))
    calibration_reps = set(map(int, cfg["training"]["lobo_calibration_repetitions_per_condition"]))
    batch_size = int(cfg["training"]["batch_size"])
    with h5py.File(cache_path, "r") as h5:
        if h5.attrs["protocol_sha256"] != protocol["protocol_sha256"]:
            raise RuntimeError("Cache protocol hash mismatch")
        for fold_index in range(args.fold_start, min(args.fold_end, len(backgrounds))):
            heldout = backgrounds[fold_index]
            train_mask = (metadata.background_id != heldout) & metadata.repetition.isin(train_reps)
            cal_mask = (metadata.background_id != heldout) & metadata.repetition.isin(calibration_reps)
            held_mask = metadata.background_id == heldout
            train_idx = metadata.loc[train_mask, "cache_index"].to_numpy(int)
            cal_idx = metadata.loc[cal_mask, "cache_index"].to_numpy(int)
            held_idx = metadata.loc[held_mask, "cache_index"].to_numpy(int)
            if set(train_idx) & set(cal_idx):
                raise RuntimeError("Model-fit/qhat leakage")
            train_data = read_indices(h5, train_idx)
            cal_data = read_indices(h5, cal_idx)
            held_data = read_indices(h5, held_idx)
            train_groups = metadata.loc[train_mask].sort_values("cache_index").background_id.map({bg: i for i, bg in enumerate(backgrounds)}).to_numpy(int)
            cal_meta = metadata.loc[cal_mask].sort_values("cache_index").reset_index(drop=True)
            held_meta = metadata.loc[held_mask].sort_values("cache_index").reset_index(drop=True)

            for candidate_id in args.candidates:
                if (candidate_id, heldout) in completed:
                    print(f"skip completed {candidate_id} {heldout}", flush=True)
                    continue
                started = time.time()
                seed = stable_seed(int(cfg["training"]["training_seed"]), heldout, candidate_id)
                model, history = fit_model(cfg, candidates[candidate_id], train_data, train_groups, seed)
                print(f"{candidate_id} fold {fold_index}: calibration inference start ({len(cal_idx)} rows)", flush=True)
                cal_pred = predict(model, cal_data[0], cal_data[1], batch_size)
                print(f"{candidate_id} fold {fold_index}: heldout inference start ({len(held_idx)} rows)", flush=True)
                held_pred = predict(model, held_data[0], held_data[1], batch_size)
                print(f"{candidate_id} fold {fold_index}: inference complete", flush=True)
                cal_score = np.maximum(
                    (cal_pred["mu"] - cal_data[2]) / np.maximum(cal_pred["s_minus"], 1e-6),
                    (cal_data[2] - cal_pred["mu"]) / np.maximum(cal_pred["s_plus"], 1e-6),
                )
                group_q = []
                for bg in sorted(cal_meta.background_id.unique()):
                    group_q.append(finite_quantile(cal_score[cal_meta.background_id.to_numpy() == bg], 0.90))
                qhat = float(max(group_q))
                print(f"{candidate_id} fold {fold_index}: qhat={qhat:.6f}", flush=True)
                lower_raw = held_pred["mu"] - qhat * held_pred["s_minus"]
                upper = held_pred["mu"] + qhat * held_pred["s_plus"]
                lower = np.maximum(0.0, lower_raw)
                covered = (held_data[2] >= lower) & (held_data[2] <= upper)
                pred_frame = held_meta.copy()
                for key, value in held_pred.items():
                    pred_frame[key] = value
                pred_frame["candidate_id"] = candidate_id
                pred_frame["fold_index"] = fold_index
                pred_frame["qhat_group_guard"] = qhat
                pred_frame["lower90"] = lower
                pred_frame["upper90"] = upper
                pred_frame["width90"] = upper - lower
                pred_frame["covered90"] = covered.astype(int)
                metrics = {
                    "candidate_id": candidate_id, "fold_index": fold_index, "heldout_background": heldout,
                    "training_groups": 35, "model_fit_rows": len(train_idx), "qhat_rows": len(cal_idx),
                    "heldout_rows": len(held_idx), "qhat_group_guard": qhat,
                    **metrics_for_background(pred_frame), "runtime_seconds": time.time() - started,
                    "protocol_sha256": protocol["protocol_sha256"],
                }
                pd.DataFrame([metrics]).to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False)
                pred_frame.to_csv(predictions_path, mode="a", header=not predictions_path.exists(), index=False)
                hist = pd.DataFrame(history)
                hist.insert(0, "heldout_background", heldout)
                hist.insert(0, "fold_index", fold_index)
                hist.insert(0, "candidate_id", candidate_id)
                hist.to_csv(history_path, mode="a", header=not history_path.exists(), index=False)
                if candidate_id == cfg["ablations"]["primary_candidate"]:
                    print(f"{candidate_id} fold {fold_index}: saving weights", flush=True)
                    model.save_weights(output / "checkpoints" / f"fold_{fold_index:02d}_{candidate_id}.h5")
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)

    print("WACQNet requested LOBO folds complete", flush=True)


if __name__ == "__main__":
    main()

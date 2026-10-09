"""Reliability-boundary primitives with explicit condition/realization semantics."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


def reliability_labels(true_emission: np.ndarray, predicted_emission: np.ndarray,
                       thresholds: Iterable[float]=(0.20,0.30,0.40)) -> pd.DataFrame:
    """Return positive-emission APE and thresholded reliability labels."""
    truth=np.asarray(true_emission,float);prediction=np.asarray(predicted_emission,float)
    if np.any(truth<=0): raise ValueError("Reliability APE is defined only for Q>0")
    ape=np.abs(prediction-truth)/truth
    result={"ape":ape,"absolute_error":np.abs(prediction-truth),"bias":prediction-truth,
            "signed_percentage_error":(prediction-truth)/truth,
            "stabilized_relative_error":np.abs(prediction-truth)/(truth+1.0)}
    for threshold in thresholds: result[f"reliable_{int(round(100*threshold))}"]=(ape<threshold).astype(np.int8)
    return pd.DataFrame(result)


def aggregate_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated realizations without treating them as independent conditions."""
    required={"condition_id","background_id","true_emission","predicted_emission","ape","operational_detected","reliable_20","reliable_30","reliable_40"}
    missing=required-set(frame)
    if missing: raise KeyError(f"Missing reliability fields: {sorted(missing)}")
    metadata=[c for c in ("experiment","evaluation_scope","split","level","background_id","background_type","true_emission",
        "true_wind_speed","input_wind_speed","wind_speed_multiplier","wind_direction_error","wind_experiment_type",
        "background_sigma","background_complexity_bin","valid_fraction","retained_valid_fraction","missingness_type","p_det","hard_class") if c in frame]
    agg=frame.groupby("condition_id",as_index=False).agg(
        n_realizations=("ape","size"),p_detectable=("operational_detected","mean"),
        p_reliable_20=("reliable_20","mean"),p_reliable_30=("reliable_30","mean"),p_reliable_40=("reliable_40","mean"),
        p_joint=("joint_reliable_30","mean"),median_ape=("ape","median"),q1_ape=("ape",lambda x:x.quantile(.25)),
        q3_ape=("ape",lambda x:x.quantile(.75)),p90_ape=("ape",lambda x:x.quantile(.90)),
        mae=("absolute_error","mean"),bias=("bias","mean"),median_signed_percentage_error=("signed_percentage_error","median"),
    )
    first=frame.groupby("condition_id",as_index=False)[metadata].first()
    agg=first.merge(agg,on="condition_id",how="left",validate="one_to_one")
    agg["p_reliable_se_20"]=np.sqrt(agg.p_reliable_20*(1-agg.p_reliable_20)/agg.n_realizations)
    agg["p_reliable_se_30"]=np.sqrt(agg.p_reliable_30*(1-agg.p_reliable_30)/agg.n_realizations)
    agg["p_reliable_se_40"]=np.sqrt(agg.p_reliable_40*(1-agg.p_reliable_40)/agg.n_realizations)
    return agg


def rotate_wind(u: np.ndarray,v: np.ndarray,error_deg: float,multiplier: float=1.0)->tuple[np.ndarray,np.ndarray]:
    """Rotate and scale an input wind field while leaving XCO2/plume unchanged."""
    angle=np.deg2rad(float(error_deg));c,s=np.cos(angle),np.sin(angle)
    return (multiplier*(u*c-v*s)).astype(np.float32),(multiplier*(u*s+v*c)).astype(np.float32)


def block_correlated_mask(valid: np.ndarray, retained_fraction: float, block_size: int,
                          scene_key: str, seed: int) -> np.ndarray:
    """Remove deterministic spatial blocks; never inspect plume or model output."""
    valid=np.asarray(valid,bool);target=max(1,int(round(valid.sum()*float(retained_fraction))))
    blocks=[]
    for y in range(0,valid.shape[0],block_size):
        for x in range(0,valid.shape[1],block_size):
            token=hashlib.sha256(f"{seed}|{scene_key}|{y}|{x}".encode()).hexdigest();blocks.append((token,y,x))
    output=valid.copy()
    for _,y,x in sorted(blocks):
        if output.sum()<=target: break
        block=np.zeros_like(output);block[y:y+block_size,x:x+block_size]=True
        removable=np.flatnonzero(output&block)
        excess=int(output.sum()-target)
        if len(removable)<=excess: output.flat[removable]=False
        elif excess>0: output.flat[removable[:excess]]=False
    return output


def operational_downwind_mask(valid: np.ndarray, operational_template: np.ndarray,
                              retained_fraction: float) -> np.ndarray:
    """Remove highest operational-template support without any true plume mask."""
    valid=np.asarray(valid,bool);template=np.abs(np.asarray(operational_template,float));target=max(1,int(round(valid.sum()*float(retained_fraction))))
    output=valid.copy();candidates=np.flatnonzero(valid);order=candidates[np.argsort(template.flat[candidates],kind="stable")[::-1]]
    output.flat[order[:max(0,int(valid.sum()-target))]]=False
    return output


def background_cluster_bootstrap(frame: pd.DataFrame, value: str, repetitions: int=1000, seed: int=42) -> dict[str,float]:
    """Bootstrap by background_id, never by derived realization."""
    if "background_id" not in frame: raise KeyError("background_id is required")
    groups=frame.background_id.unique()
    if len(groups)<2: return {"estimate":float(frame[value].mean()),"ci_lower":float("nan"),"ci_upper":float("nan"),"n_backgrounds":len(groups)}
    rng=np.random.default_rng(seed);values=[]
    per=frame.groupby("background_id")[value].mean()
    for _ in range(repetitions): values.append(float(per.loc[rng.choice(groups,size=len(groups),replace=True)].mean()))
    return {"estimate":float(per.mean()),"ci_lower":float(np.quantile(values,.025)),"ci_upper":float(np.quantile(values,.975)),"n_backgrounds":len(groups)}

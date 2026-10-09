"""Run the frozen HPR over leakage-safe Level-B reliability-boundary grids."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.detectability_experiment import _score_image
from src.data.matched_filter import TemplateConfig,generate_operational_template
from src.data.null_normalization import deterministic_null_controls,scene_normalized_score
from src.data.preprocessing import build_residual
from src.evaluation.reliability_boundary import (
    aggregate_conditions,block_correlated_mask,operational_downwind_mask,
    reliability_labels,rotate_wind,
)
from src.evaluation.zero_plume import fill_per_channel
from src.models.backbones import build_legacy_model


def sha_array(value:np.ndarray)->str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


class FrozenHPR:
    """Exact second-paper preprocessing/model wrapper; it never fits statistics."""
    def __init__(self,cfg:dict[str,Any],output:Path):
        b=cfg["backbone"]
        loaded=build_legacy_model(b["legacy_repo"],b["name"],tuple(b["input_shape"]),b["checkpoint"],b["dropout_rate"],output/"checkpoint_load_report.json",b["normalization_layer"])
        loaded.backbone.freeze_all();self.model=loaded.full_model
        self.fill=np.asarray(json.loads(Path(cfg["input"]["legacy_nan_fill"]).read_text(encoding="utf-8"))["values"],np.float32)
        self.checkpoint=b["checkpoint"]
    def predict(self,images:list[np.ndarray]|np.ndarray,batch_size:int=256)->np.ndarray:
        array=fill_per_channel(np.asarray(images,np.float32),self.fill)
        values=[]
        for start in range(0,len(array),batch_size): values.append(np.asarray(self.model(array[start:start+batch_size],training=False)).reshape(-1))
        return np.concatenate(values).astype(np.float32)


class FrozenOperationalDetector:
    """Frozen v2 detector; tau/template/null parameters are read-only inputs."""
    def __init__(self,cfg:dict[str,Any]):
        self.det=yaml.safe_load(Path(cfg["input"]["detector_config"]).read_text(encoding="utf-8"))["detector"]
        calibration=json.loads(Path(cfg["input"]["detector_calibration"]).read_text(encoding="utf-8"));self.tau=float(calibration["tau0_z"])
        self.template_cfg=TemplateConfig(**self.det["template"]);self.seed=int(cfg["seed"]);self.cache={}
    def score(self,image:np.ndarray,valid:np.ndarray,scene_key:str)->dict[str,Any]:
        u=image[...,1];v=image[...,2];uu=float(np.nanmedian(u));vv=float(np.nanmedian(v))
        raw,quality,residual=_score_image(image[...,0],valid,u,v,(self.det["source_y"],self.det["source_x"]),{"matched_filter":self.det},self.template_cfg)
        key=(scene_key,sha_array(valid.astype(np.uint8)),round(uu,5),round(vv,5))
        if key not in self.cache:
            self.cache[key]=deterministic_null_controls(valid,(self.det["source_y"],self.det["source_x"]),uu,vv,self.det["grid_resolution_km"],self.template_cfg,scene_key,self.seed,self.det["k_null"],self.det["min_source_distance_pixels"],self.det["max_template_overlap"],self.det["plume_exclusion_sigma"],self.det["source_buffer_km"],self.det["min_background_pixels"])
        z,info=scene_normalized_score(residual,raw,self.cache[key][0],self.det["variance_estimator"],self.det["min_background_pixels"],self.det["epsilon"])
        return {"matched_filter_score_raw":raw,"operational_z_score":z,"operational_detected":int(z>self.tau),
                "background_sigma":quality["sigma_background"],"residual_std":quality["residual_std"],
                "valid_fraction":float(valid.mean()),"detector_quality_flag":quality["quality_flag"],
                "null_quality_flag":info["null_quality_flag"]}


def input_direction(u:np.ndarray,v:np.ndarray)->float:
    return float(np.degrees(np.arctan2(np.nanmedian(v),np.nanmedian(u))))


def baseline_rows(frame:pd.DataFrame,h5:h5py.File,hpr:FrozenHPR,tau:float)->pd.DataFrame:
    positive=frame[(frame.emission>0)&(frame.split!="calibration")].copy().reset_index(drop=True)
    predictions=[]
    for start in range(0,len(positive),256):
        indices=positive.row_index.iloc[start:start+256].to_numpy(int);predictions.extend(hpr.predict(np.asarray(h5["image"][indices],np.float32)).tolist())
        if start%2048==0: print(f"baseline inference {start}/{len(positive)}",flush=True)
    positive["true_emission"]=positive.emission;positive["predicted_emission"]=predictions
    positive["experiment"]="physical_wind";positive["evaluation_scope"]=np.where(positive.split=="test","frozen_external_test","development")
    positive["level"]="B";positive["true_wind_speed"]=positive.wind_speed;positive["input_wind_speed"]=positive.wind_speed
    positive["wind_speed_multiplier"]=1.0;positive["wind_speed_error"]=0.0;positive["true_wind_direction"]=positive.wind_direction
    positive["input_wind_direction"]=positive.wind_direction;positive["wind_direction_error"]=0.0
    positive["wind_experiment_type"]="physical_simulation_with_matched_plume_and_uv"
    positive["retained_valid_fraction"]=1.0;positive["missingness_type"]="natural_oco3_mask"
    positive["operational_detected"]=(positive.operational_z_score>tau).astype(np.int8)
    return positive


def intervention_design(frame:pd.DataFrame,cfg:dict[str,Any])->pd.DataFrame:
    d=cfg["design"];base=frame[(frame.emission.isin(d["core_emissions_mt_yr"]))&(frame.source_library_row==d["base_physical_wind_row"])&(frame.split!="calibration")].copy()
    rows=[]
    for row in base.itertuples(index=False):
        for error in d["wind_direction_errors_deg"]:
            if float(error)==0: continue
            rows.append({"source_row_index":int(row.row_index),"experiment":"wind_direction_error","parameter":float(error),"missingness_type":"none"})
        for multiplier in d["wind_speed_multipliers"]:
            if float(multiplier)==1: continue
            rows.append({"source_row_index":int(row.row_index),"experiment":"wind_speed_input_error","parameter":float(multiplier),"missingness_type":"none"})
        for kind in d["missingness_types"]:
            for retained in d["retained_valid_fractions"]:
                if float(retained)==1: continue
                rows.append({"source_row_index":int(row.row_index),"experiment":"missingness","parameter":float(retained),"missingness_type":kind})
    return pd.DataFrame(rows)


def build_interventions(design:pd.DataFrame,frame:pd.DataFrame,h5:h5py.File,cfg:dict[str,Any],hpr:FrozenHPR,detector:FrozenOperationalDetector)->pd.DataFrame:
    source=frame.set_index("row_index");records=[];images=[];metadata=[];d=cfg["design"]
    for number,item in enumerate(design.itertuples(index=False)):
        row=source.loc[item.source_row_index];image=np.asarray(h5["image"][item.source_row_index],np.float32);valid=np.asarray(h5["valid_mask"][item.source_row_index],bool)
        true_u=image[...,1].copy();true_v=image[...,2].copy();input_u=true_u;input_v=true_v;new_valid=valid.copy()
        error=0.0;multiplier=1.0;retained=1.0
        if item.experiment=="wind_direction_error": error=float(item.parameter);input_u,input_v=rotate_wind(true_u,true_v,error,1.0)
        elif item.experiment=="wind_speed_input_error": multiplier=float(item.parameter);input_u,input_v=rotate_wind(true_u,true_v,0.0,multiplier)
        else:
            retained=float(item.parameter);scene_key=f"{row.background_id}|{row.condition_id}|r={int(row.repetition)}|{item.missingness_type}|{retained}"
            if item.missingness_type=="block_correlated": new_valid=block_correlated_mask(valid,retained,int(d["missingness_block_size_px"]),scene_key,int(cfg["seed"]))
            else:
                uu=float(np.nanmedian(true_u));vv=float(np.nanmedian(true_v));template,_=generate_operational_template(valid.shape,(15.5,15.5),uu,vv,2.0,valid,detector.template_cfg)
                new_valid=operational_downwind_mask(valid,template,retained)
            image[...,0]=np.where(new_valid,image[...,0],np.nan)
        image[...,1]=input_u;image[...,2]=input_v
        condition=f"{row.background_id}|Q={row.emission:g}|windrow={int(row.source_library_row)}|{item.experiment}|parameter={item.parameter:g}|missing={item.missingness_type}"
        info=detector.score(image,new_valid,condition)
        metadata.append({"source_row_index":item.source_row_index,"sample_id":f"{condition}|r={int(row.repetition):03d}","condition_id":condition,
            "background_id":row.background_id,"background_type":row.background_type,"split":row.split,"evaluation_scope":"frozen_external_test" if row.split=="test" else "development","level":"B",
            "experiment":item.experiment,"true_emission":float(row.emission),"true_wind_speed":float(row.wind_speed),"input_wind_speed":float(np.hypot(np.nanmedian(input_u),np.nanmedian(input_v))),
            "wind_speed_multiplier":multiplier,"wind_speed_error":float(np.hypot(np.nanmedian(input_u),np.nanmedian(input_v))-row.wind_speed),
            "true_wind_direction":float(row.wind_direction),"input_wind_direction":input_direction(input_u,input_v),"wind_direction_error":error,
            "wind_experiment_type":"operational_input_error_fixed_xco2_plume" if item.experiment.startswith("wind_") else "fixed_physical_plume_mask_perturbation",
            "retained_valid_fraction":retained,"missingness_type":item.missingness_type,"repetition":int(row.repetition),"n_repetitions":int(d["repetitions"]),
            "reference_emission":row.reference_emission,"plume_scaling_factor":row.plume_scaling_factor,"source_library_row":row.source_library_row,**info})
        images.append(image)
        if len(images)==256:
            prediction=hpr.predict(images)
            for meta,value in zip(metadata,prediction): meta["predicted_emission"]=float(value);records.append(meta)
            images=[];metadata=[]
        if (number+1)%2000==0: print(f"interventions {number+1}/{len(design)}",flush=True)
    if images:
        prediction=hpr.predict(images)
        for meta,value in zip(metadata,prediction): meta["predicted_emission"]=float(value);records.append(meta)
    return pd.DataFrame(records)


def add_reliability(frame:pd.DataFrame,thresholds:list[float])->pd.DataFrame:
    labels=reliability_labels(frame.true_emission,frame.predicted_emission,thresholds)
    result=pd.concat([frame.reset_index(drop=True),labels],axis=1);result["joint_reliable_30"]=(result.operational_detected.astype(bool)&result.reliable_30.astype(bool)).astype(np.int8)
    result["seed"]=42
    return result


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out=Path(cfg["output_dir"]);out.mkdir(parents=True,exist_ok=True);(out/"figures").mkdir(exist_ok=True);shutil.copy2(args.config,out/"config.yaml")
    frame=pd.read_csv(cfg["input"]["manifest"]);calibration=frame[frame.split==cfg["design"]["sealed_calibration_split"]]
    detector=FrozenOperationalDetector(cfg);hpr=FrozenHPR(cfg,out)
    with h5py.File(cfg["input"]["h5"],"r") as h5:
        baseline=baseline_rows(frame,h5,hpr,detector.tau)
        design=intervention_design(frame,cfg);design.to_csv(out/"intervention_design.csv",index=False)
        interventions=build_interventions(design,frame,h5,cfg,hpr,detector)
    baseline_columns=[c for c in interventions.columns if c in baseline.columns]
    missing=[c for c in interventions.columns if c not in baseline]
    for c in missing: baseline[c]=np.nan
    baseline=baseline[interventions.columns]
    realizations=add_reliability(pd.concat([baseline,interventions],ignore_index=True),cfg["reliability"]["ape_thresholds"])
    # P_det and hard class are condition properties, assigned only after all R realizations exist.
    realizations["p_det"]=realizations.groupby("condition_id").operational_detected.transform("mean")
    realizations["hard_class"]=np.where(realizations.p_det<float(cfg["reliability"]["quantifiable_pdet_threshold"]),1,2)
    dev=realizations.evaluation_scope=="development"
    baseline_complexity=baseline.groupby("background_id").background_sigma.median()
    complexity=baseline[baseline.evaluation_scope=="development"].groupby("background_id").background_sigma.median();cuts=np.quantile(complexity,cfg["design"]["background_complexity_quantiles"])
    realizations["background_complexity_reference"]=realizations.background_id.map(baseline_complexity)
    realizations["background_complexity_bin"]=pd.cut(realizations.background_complexity_reference,[-np.inf,cuts[0],cuts[1],np.inf],labels=["low","medium","high"],include_lowest=True).astype(str)
    realizations.to_csv(out/"reliability_realizations.csv",index=False)
    conditions=aggregate_conditions(realizations);conditions.to_csv(out/"reliability_conditions.csv",index=False)
    audit={"scientific_scope":"Formal boundaries are Level B: real OCO-3 background plus physically matched SMARTCARB plume/u/v.",
        "level_a_status":"Not constructed: the current SMARTCARB asset lacks a verified absolute-XCO2 background input compatible with the frozen HPR.",
        "n_backgrounds":int(realizations.background_id.nunique()),"n_development_backgrounds":int(realizations.loc[dev,"background_id"].nunique()),"n_external_test_backgrounds":int(realizations.loc[~dev,"background_id"].nunique()),
        "sealed_calibration_backgrounds":sorted(calibration.background_id.unique().tolist()),"sealed_calibration_h5_rows_read":0,
        "n_realizations":len(realizations),"n_conditions":len(conditions),"q_range":[float(realizations.true_emission.min()),float(realizations.true_emission.max())],
        "physical_wind_range":[float(baseline.true_wind_speed.min()),float(baseline.true_wind_speed.max())],"valid_fraction_range":[float(realizations.valid_fraction.min()),float(realizations.valid_fraction.max())],
        "background_complexity_cutpoints_train_validation_only":cuts.tolist(),"background_bin_fit_splits":cfg["design"]["development_splits"],
        "tau0_z_frozen":detector.tau,"normalization_refit":False,"checkpoint":hpr.checkpoint}
    (out/"reliability_dataset_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    (out/"reliability_dataset_audit.md").write_text("# Reliability Dataset Audit\n\n```json\n"+json.dumps(audit,indent=2)+"\n```\n",encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main()

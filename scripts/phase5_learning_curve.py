
from __future__ import annotations
import json, time, random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loaders import LocalCsvCfg, load_local_csv
from src.runner.train_eval_sklearn import fit_and_eval

BEST_FILE = Path("results/phase2_best_global_configs.json")

OUT_DIR = Path("results/learning_curve")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS: Dict[str, Dict[str, Any]] = {
    "sms":   {"name":"sms",   "files":{"train":"data/local/sms_spam.csv"},                 "columns":{"text":"sms","label":"label"},   "val_size":0.1, "test_size":0.2},
    "imdb":  {"name":"imdb",  "files":{"train":"data/local/imdb_train.csv","test":"data/local/imdb_test.csv"}, "columns":{"text":"text","label":"label"}, "val_size":0.1},
}

FRACTIONS: List[float] = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

SEEDS = [101, 102, 103]

def _read_best_configs(path: Path):

    data = json.loads(path.read_text(encoding="utf-8"))

    out = []
    for row in data:
        model = row.get("model")
        cfg   = row.get("model_cfg")
        if model is None or cfg is None:
            continue
        out.append({"model": model, "cfg": cfg})
    return out

def _load(ds_cfg: Dict[str,Any], split: str, seed: int):
    lcfg = LocalCsvCfg(
        files=ds_cfg["files"],
        columns=ds_cfg["columns"],
        val_size=ds_cfg["val_size"],
        test_size=ds_cfg.get("test_size"),
        stratify=True, sep=",", encoding="utf-8",
        random_state=seed,
    )
    return load_local_csv(split, lcfg)

def _class_dist(series: pd.Series) -> Dict[str,int]:
    vc = series.value_counts().to_dict()
    return {str(k): int(v) for k, v in vc.items()}

def _subsample_train_stratified(train_df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    frac = float(frac)
    if frac >= 0.999:
        return train_df

    y = train_df["label"]
    min_per_class = 2
    cls_counts = y.value_counts()
    adj_frac = frac
    for c, cnt in cls_counts.items():
        need = min_per_class / cnt
        if need > adj_frac:
            adj_frac = need
    adj_frac = min(adj_frac, 1.0)
    df_small, _ = train_test_split(train_df, train_size=adj_frac, stratify=y, random_state=seed)
    return df_small

def main():
    best_cfgs = _read_best_configs(BEST_FILE)

    for ds_name in ["sms", "imdb"]:
        ds = DATASETS[ds_name]
        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed)

            full_train = _load(ds, "train", seed)
            val_df     = _load(ds, "validation", seed)
            test_df    = _load(ds, "test", seed)

            for frac in FRACTIONS:
                train_sub = _subsample_train_stratified(full_train, frac, seed)
                dist_train = _class_dist(train_sub["label"])
                dist_val   = _class_dist(val_df["label"])
                dist_test  = _class_dist(test_df["label"])

                for row in best_cfgs:
                    model_name = row["model"]
                    model_cfg  = row["cfg"]

                    t0 = time.perf_counter()
                    res = fit_and_eval(model_name, model_cfg, train_sub, {"val": val_df, "test": test_df}, seed)
                    elapsed = time.perf_counter() - t0

                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"{model_name}_{ds_name}_frac{int(frac*100)}_s{seed}_{stamp}.json"
                    payload = {
                        "meta": {
                            "phase": "phase5_learning_curve",
                            "timestamp": stamp,
                            "dataset": ds_name,
                            "model": model_name,
                            "model_cfg": model_cfg,
                            "seed": seed,
                            "train_frac": float(frac),
                            "train_size_used": int(len(train_sub)),
                            "val_size_used": int(len(val_df)),
                            "test_size_used": int(len(test_df)),
                            "class_dist_train": dist_train,
                            "class_dist_val": dist_val,
                            "class_dist_test": dist_test,
                        },
                        "metrics": {
                            **res["test"],
                            "elapsed_sec": float(elapsed),
                        },
                        "val_metrics": res.get("val", {}),
                    }
                    (OUT_DIR / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"saved {OUT_DIR/fname}")

    print("\nPhase 5  finished.", OUT_DIR)

if __name__ == "__main__":
    main()

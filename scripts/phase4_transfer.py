
from __future__ import annotations
import json, time, random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loaders import LocalCsvCfg, load_local_csv
from src.runner.train_eval_sklearn import fit_and_eval


BEST_FILE = Path("results/phase2_best_global_configs.json")


OUT_DIR = Path("results/transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "sms":    {"name":"sms",   "files":{"train":"data/local/sms_spam.csv"},                 "columns":{"text":"sms","label":"label"},   "val_size":0.1, "test_size":0.2},
    "enron":  {"name":"enron", "files":{"train":"data/local/enron_spam.csv"},               "columns":{"text":"email","label":"label"}, "val_size":0.1, "test_size":0.2},
    "imdb":   {"name":"imdb",  "files":{"train":"data/local/imdb_train.csv","test":"data/local/imdb_test.csv"},   "columns":{"text":"text","label":"label"},            "val_size":0.1},
    "amazon": {"name":"amazon","files":{"train":"data/local/amazon_train.csv","test":"data/local/amazon_test.csv"},"columns":{"text_cols":["title","content"],"label":"label"},"val_size":0.1},
}


PAIRS = [
    ("sms","enron"),
    ("enron","sms"),
    ("imdb","amazon"),
    ("amazon","imdb"),
]


SEEDS = [101, 102, 103]


def _load(ds_cfg: Dict[str,Any], split: str, seed: int):
    lcfg = LocalCsvCfg(
        files=ds_cfg["files"],
        columns=ds_cfg["columns"],
        val_size=ds_cfg["val_size"],
        test_size=ds_cfg.get("test_size"),
        stratify=True,
        sep=",",
        encoding="utf-8",
        random_state=seed,
    )
    return load_local_csv(split, lcfg)

def _class_dist(series: pd.Series) -> Dict[str,int]:
    vc = series.value_counts().to_dict()
    return {str(k): int(v) for k, v in vc.items()}

def _stratified_subsample_df(df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    if target_n >= len(df):
        return df
    y = df["label"]

    frac = target_n / len(df)
    df_small, _ = train_test_split(df, train_size=frac, stratify=y, random_state=seed)
    return df_small

def _read_best_configs(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))

    best = []
    for row in data:
        model = row.get("model")
        cfg   = row.get("model_cfg")
        if model is None or cfg is None:
            continue
        best.append({"model": model, "cfg": cfg})

    return best

def _cap_for_pair(train_src_len: int, train_tgt_len: int) -> int:
    return min(int(train_src_len), int(train_tgt_len))

def main():

    best_cfg_list = _read_best_configs(BEST_FILE)

    for (src_name, tgt_name) in PAIRS:
        src = DATASETS[src_name]
        tgt = DATASETS[tgt_name]

        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed)

            train_src = _load(src, "train", seed)
            train_tgt = _load(tgt, "train", seed)
            test_tgt  = _load(tgt, "test",  seed)

            cap = _cap_for_pair(len(train_src), len(train_tgt))
            train_src_used = _stratified_subsample_df(train_src, cap, seed)

            dist_src = _class_dist(train_src_used["label"])
            dist_tgt = _class_dist(test_tgt["label"])

            for row in best_cfg_list:
                model_name = row["model"]
                model_cfg  = row["cfg"]

                t0 = time.perf_counter()
                res = fit_and_eval(model_name, model_cfg, train_src_used, {"test": test_tgt}, seed)
                elapsed = time.perf_counter() - t0

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{model_name}_train{src_name}_test{tgt_name}_size_matched_s{seed}_{stamp}.json"

                payload = {
                    "meta": {
                        "phase": "phase4_transfer",
                        "timestamp": stamp,
                        "regime": "size_matched",
                        "train_dataset": src_name,
                        "test_dataset": tgt_name,
                        "model": model_name,
                        "model_cfg": model_cfg,
                        "seed": seed,
                        "train_size_used": int(len(train_src_used)),
                        "test_size_used": int(len(test_tgt)),
                        "class_dist_src": dist_src,
                        "class_dist_tgt": dist_tgt,
                    },
                    "metrics": {  
                        **res["test"],
                        "elapsed_sec": float(elapsed),
                    },
                }

                (OUT_DIR / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"saved {OUT_DIR/fname}")

    print("\nPhase 4 finished.", OUT_DIR)

if __name__ == "__main__":
    main()

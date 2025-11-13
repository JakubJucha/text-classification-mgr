from __future__ import annotations
import json, time, random
from pathlib import Path
from datetime import datetime
import numpy as np

from src.data.loaders import LocalCsvCfg, load_local_csv
from src.runner.train_eval_sklearn import run_sklearn

BEST_FILE = Path("results/phase2_best_global_configs.json")

OUT_DIR = Path("results/rebaseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {"name":"sms",   "files":{"train":"data/local/sms_spam.csv"},                 "columns":{"text":"sms","label":"label"},   "val_size":0.1, "test_size":0.2},
    {"name":"enron", "files":{"train":"data/local/enron_spam.csv"},               "columns":{"text":"email","label":"label"}, "val_size":0.1, "test_size":0.2},
    {"name":"imdb",  "files":{"train":"data/local/imdb_train.csv","test":"data/local/imdb_test.csv"},   "columns":{"text":"text","label":"label"},            "val_size":0.1},
    {"name":"amazon","files":{"train":"data/local/amazon_train.csv","test":"data/local/amazon_test.csv"},"columns":{"text_cols":["title","content"],"label":"label"},"val_size":0.1},
]

SEEDS = [101, 102, 103]

def _load(ds, split:str, seed:int):
    lcfg = LocalCsvCfg(
        files=ds["files"], columns=ds["columns"],
        val_size=ds["val_size"], test_size=ds.get("test_size"),
        stratify=True, sep=",", encoding="utf-8",
        random_state=seed,
    )
    return load_local_csv(split, lcfg)

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

def main():
    best_cfg_list = _read_best_configs(BEST_FILE)

    for row in best_cfg_list:
        model_name = row["model"]
        model_cfg  = row["cfg"]

        for ds in DATASETS:
            ds_name = ds["name"]
            for seed in SEEDS:
                random.seed(seed); np.random.seed(seed)

                train_df = _load(ds, "train", seed)
                val_df   = _load(ds, "validation", seed)
                test_df  = _load(ds, "test", seed)

                t0 = time.perf_counter()
                metrics = run_sklearn(model_name, model_cfg, train_df, val_df, test_df, seed)
                metrics["elapsed_sec"] = time.perf_counter() - t0

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{model_name}_{ds_name}_s{seed}_{stamp}.json"
                payload = {
                    "meta": {
                        "phase": "phase3_rebaseline",
                        "timestamp": stamp,
                        "dataset": ds_name,
                        "files": ds["files"],
                        "columns": ds["columns"],
                        "model": model_name,
                        "model_cfg": model_cfg,
                        "seed": seed,
                    },
                    "metrics": metrics, 
                }
                (OUT_DIR / fname).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"saved {OUT_DIR/fname}")

if __name__ == "__main__":
    main()

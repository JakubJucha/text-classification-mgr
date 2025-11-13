from __future__ import annotations
import json, time, random
from pathlib import Path
from datetime import datetime
import numpy as np

from src.data.loaders import LocalCsvCfg, load_local_csv
from src.runner.train_eval_sklearn import run_sklearn

DATASETS = [
    {"name":"sms",   "files":{"train":"data/local/sms_spam.csv"},                 "columns":{"text":"sms","label":"label"},   "val_size":0.1, "test_size":0.2},
    {"name":"enron", "files":{"train":"data/local/enron_spam.csv"},               "columns":{"text":"email","label":"label"}, "val_size":0.1, "test_size":0.2},
    {"name":"imdb",  "files":{"train":"data/local/imdb_train.csv","test":"data/local/imdb_test.csv"},   "columns":{"text":"text","label":"label"},            "val_size":0.1},
    {"name":"amazon","files":{"train":"data/local/amazon_train.csv","test":"data/local/amazon_test.csv"},"columns":{"text_cols":["title","content"],"label":"label"},"val_size":0.1},
]

TFIDF = {"ngram_min":1, "ngram_max":2, "min_df":2, "max_features":100_000}

MODELS = [
    ("logreg", {"C":1.0, "class_weight":None, "tfidf":TFIDF}),
    ("nb",     {"alpha":0.5, "tfidf":TFIDF}),
    ("svm",    {"C":1.0, "class_weight":None, "tfidf":TFIDF}),
    ("rf",     {"n_estimators":200, "max_depth":None, "tfidf":TFIDF}),
]

SEEDS = [101, 102, 103]

OUT_DIR = Path("results/baselines")

def _load(ds, split: str, seed: int):
    lcfg = LocalCsvCfg(
        files=ds["files"],
        columns=ds["columns"],
        val_size=ds["val_size"],
        test_size=ds.get("test_size"),
        stratify=True,
        sep=",",
        encoding="utf-8",
        random_state=seed,  
    )
    return load_local_csv(split, lcfg)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    ensure_dir(OUT_DIR)

    for ds in DATASETS:
        ds_name = ds["name"]

        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed)

            train_df = _load(ds, "train", seed)
            val_df   = _load(ds, "validation", seed)
            test_df  = _load(ds, "test", seed)

            for model_name, model_cfg in MODELS:
                t0 = time.perf_counter()
                metrics = dict(run_sklearn(model_name, model_cfg, train_df, val_df, test_df, seed=seed))
                metrics["elapsed_sec"] = time.perf_counter() - t0

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = f"{model_name}_{ds_name}_s{seed}_{stamp}.json"
                payload = {
                    "meta": {
                        "phase": "phase1_baseline",
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
                (OUT_DIR / out_name).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"saved {OUT_DIR/out_name}")

if __name__ == "__main__":
    main()

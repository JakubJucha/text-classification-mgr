from __future__ import annotations
import json, time, random
from pathlib import Path
from datetime import datetime

import numpy as np
from src.data.loaders import LocalCsvCfg, load_local_csv
from src.runner.train_eval_sklearn import fit_and_eval


DATASETS = [
    {"name":"sms",   "files":{"train":"data/local/sms_spam.csv"},                 "columns":{"text":"sms","label":"label"},   "val_size":0.1, "test_size":0.2},
    {"name":"imdb",  "files":{"train":"data/local/imdb_train.csv","test":"data/local/imdb_test.csv"},   "columns":{"text":"text","label":"label"},            "val_size":0.1},
]

SEEDS = [101, 102]

OUT_DIR = Path("results/tuning")  
OUT_DIR.mkdir(parents=True, exist_ok=True)


TFIDF_GRID = [
    {"ngram_min":1, "ngram_max":2, "min_df":2, "max_df":0.95,
     "max_features":100_000, "stop_words":"english", "sublinear_tf":True, "binary":False},
    {"ngram_min":1, "ngram_max":2, "min_df":2, "max_df":1.0,
     "max_features":100_000, "stop_words":None, "sublinear_tf":True, "binary":False},
    {"ngram_min":1, "ngram_max":2, "min_df":2, "max_df":0.95,
     "max_features":200_000, "stop_words":"english", "sublinear_tf":True, "binary":False},
    {"ngram_min":1, "ngram_max":2, "min_df":5, "max_df":0.9,
     "max_features":100_000, "stop_words":"english", "sublinear_tf":False, "binary":True},
]


LOGREG_GRID = [
    {"C":0.5, "class_weight":None,       "solver":"liblinear"},
    {"C":1.0, "class_weight":None,       "solver":"liblinear"},
    {"C":2.0, "class_weight":None,       "solver":"liblinear"},
    {"C":1.0, "class_weight":"balanced", "solver":"liblinear"},
    {"C":1.0, "class_weight":None,       "solver":"saga"},
]
SVM_GRID = [
    {"C":0.5, "class_weight":None},
    {"C":1.0, "class_weight":None},
    {"C":2.0, "class_weight":None},
    {"C":1.0, "class_weight":"balanced"},
]
NB_GRID = [
    {"alpha":0.1},
    {"alpha":0.5},
    {"alpha":1.0},
    {"alpha":1.5},
]
RF_GRID = [
    {"n_estimators":200, "max_depth":None, "n_jobs":-1},
    {"n_estimators":500, "max_depth":None, "n_jobs":-1},
    {"n_estimators":500, "max_depth":50,   "n_jobs":-1},
    {"n_estimators":800, "max_depth":50,   "n_jobs":-1},
]

MODELS = [
    ("logreg", LOGREG_GRID),
    ("svm",    SVM_GRID),
    ("nb",     NB_GRID),
    ("rf",     RF_GRID),
]

def _load(ds, split: str, seed: int):
    lcfg = LocalCsvCfg(
        files=ds["files"], columns=ds["columns"],
        val_size=ds["val_size"], test_size=ds.get("test_size"),
        stratify=True, sep=",", encoding="utf-8",
        random_state=seed,
    )
    return load_local_csv(split, lcfg)

def main():
    combo_id = 0  

    for model_name, model_grid in MODELS:
        for tfidf_cfg in TFIDF_GRID:
            for model_cfg in model_grid:
                combo_id += 1
                
                full_cfg = dict(model_cfg)
                full_cfg["tfidf"] = dict(tfidf_cfg)

                for ds in DATASETS:
                    ds_name = ds["name"]
                    for seed in SEEDS:
                        random.seed(seed); np.random.seed(seed)

                        
                        train_df = _load(ds, "train", seed)
                        val_df   = _load(ds, "validation", seed)
                        test_df  = _load(ds, "test", seed)

                        
                        t0 = time.perf_counter()
                        res = fit_and_eval(model_name, full_cfg, train_df, {"val": val_df, "test": test_df}, seed)
                        elapsed = time.perf_counter() - t0

                        
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_name = f"{model_name}_{ds_name}_combo{combo_id}_s{seed}_{stamp}.json"

                       
                        payload = {
                            "meta": {
                                "phase": "phase2_tuning",
                                "timestamp": stamp,
                                "dataset": ds_name,
                                "files": ds["files"],
                                "columns": ds["columns"],
                                "model": model_name,
                                "model_cfg": full_cfg,     
                                "combo_id": combo_id,
                                "seed": seed,
                                "seeds_all": SEEDS,
                            },
                            "metrics": {
                                **res["test"],
                                "elapsed_sec": elapsed
                            },
                            "val_metrics": res["val"],
                        }

                        (OUT_DIR / out_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"saved {OUT_DIR/out_name}")

    print("\nPhase 2 finished.", OUT_DIR)

if __name__ == "__main__":
    main()

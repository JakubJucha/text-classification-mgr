# phase6_cost_profile.py
from __future__ import annotations
import json, time, random, os, platform, threading
from pathlib import Path
from datetime import datetime

import numpy as np
import psutil
import joblib
from threadpoolctl import threadpool_limits

from src.data.loaders import LocalCsvCfg, load_local_csv
from src.features.text_featurizer import TfIdfFeaturizer, TfIdfConfig
from src.models.sklearn_models import build_model


DATASETS = [
    {"name":"sms",   "files":{"train":"data/local/sms_spam.csv"},                 "columns":{"text":"sms","label":"label"},   "val_size":0.1, "test_size":0.2},
    {"name":"imdb",  "files":{"train":"data/local/imdb_train.csv","test":"data/local/imdb_test.csv"}, "columns":{"text":"text","label":"label"}, "val_size":0.1},
]
SEEDS = [101, 102, 103]
OUT_DIR = Path("results/cost_profile"); OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def _load(ds, split: str, seed: int):
    lcfg = LocalCsvCfg(
        files=ds["files"], columns=ds["columns"],
        val_size=ds["val_size"], test_size=ds.get("test_size"),
        stratify=True, sep=",", encoding="utf-8", random_state=seed,
    )
    return load_local_csv(split, lcfg)

def _dump_model_size_bytes(pipeline, tmp_path: Path) -> int:
    joblib.dump(pipeline, tmp_path)
    size = tmp_path.stat().st_size
    tmp_path.unlink()
    return int(size)

def main():
    best_rows = json.loads(Path("results/phase2_best_global_configs.json")
                           .read_text(encoding="utf-8"))
    best = [(row["model"], row["model_cfg"]) for row in best_rows]

    _ = np.linalg.svd(np.random.randn(128, 32), full_matrices=False)

    for ds in DATASETS:
        ds_name = ds["name"]

        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed)

            train_df = _load(ds, "train", seed)
            test_df  = _load(ds, "test", seed)

            Xtr, ytr = train_df["text"], train_df["label"]
            Xte, yte = test_df["text"],  test_df["label"]

            for model_name, cfg in best:


                tfidf_cfg = cfg.get("tfidf", {})
                tfidf = TfIdfFeaturizer(TfIdfConfig(**tfidf_cfg))

                with threadpool_limits(1):
                    t0 = time.perf_counter()
                    Xtr_mat = tfidf.fit_transform(Xtr)
                    t_fit_tfidf = time.perf_counter() - t0

                feat_dim = int(Xtr_mat.shape[1])


                clf = build_model(model_name, cfg)

                with threadpool_limits(1):
                    t1 = time.perf_counter()
                    clf.fit(Xtr_mat, ytr)
                    t_fit_clf = time.perf_counter() - t1


                Xte_mat = tfidf.transform(Xte)

                with threadpool_limits(1):
                    t3 = time.perf_counter()
                    yhat = clf.predict(Xte_mat)
                    t_pred = time.perf_counter() - t3



                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                metrics = {
                    "accuracy": float(accuracy_score(yte, yhat)),
                    "f1_macro": float(f1_score(yte, yhat, average="macro")),
                    "precision_macro": float(precision_score(yte, yhat, average="macro")),
                    "recall_macro": float(recall_score(yte, yhat, average="macro")),
                }


                from sklearn.pipeline import Pipeline
                pipe = Pipeline([("tfidf", tfidf.vectorizer), ("clf", clf)])
                model_size = _dump_model_size_bytes(pipe, OUT_DIR / "_tmp_model.joblib")


                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{model_name}_{ds_name}_s{seed}_{stamp}.json"

                payload = {
                    "meta": {
                        "phase": "phase6_cost_profile",
                        "timestamp": stamp,
                        "dataset": ds_name,
                        "model": model_name,
                        "model_cfg": cfg,
                        "seed": seed,
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "feat_dim": feat_dim,
                        "env": {
                            "python": platform.python_version(),
                            "platform": platform.platform(),
                            "threads": 1,
                        },
                    },
                    "metrics": metrics,
                    "costs": {
                        "time_fit_tfidf_sec": float(t_fit_tfidf),
                        "time_fit_clf_sec": float(t_fit_clf),
                        "time_predict_sec": float(t_pred),
                        "time_total_train_sec": float(t_fit_tfidf + t_fit_clf),
                        "time_total_infer_sec": float(t_pred),
                        "model_size_bytes": int(model_size),
                    },
                }

                (OUT_DIR / fname).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print("saved", OUT_DIR / fname)

    print("\nPhase 6 finished.")


if __name__ == "__main__":
    main()
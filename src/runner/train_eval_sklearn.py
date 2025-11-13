from __future__ import annotations
from typing import Dict, Any
import numpy as np
from src.features.text_featurizer import TfIdfFeaturizer, TfIdfConfig
from src.models.sklearn_models import build_model
from src.eval.metrics import classification_metrics


def run_sklearn(model_name: str, model_cfg: Dict[str, Any], train_df, val_df, test_df, seed):
    tfidf = TfIdfFeaturizer(TfIdfConfig(**model_cfg.get("tfidf", {})))

    X_train = tfidf.fit_transform(train_df["text"])
    y_train = train_df["label"].values

    clf = build_model(model_name, model_cfg, seed)

    clf.fit(X_train, y_train)

    X_test = tfidf.transform(test_df["text"])
    y_test = test_df["label"].values
    y_pred = clf.predict(X_test)

    return classification_metrics(y_test, y_pred)

def fit_and_eval(model_name: str, model_cfg: Dict[str, Any], train_df, eval_dfs: Dict[str, Any], seed: int):
    np.random.seed(seed)
    tfidf = TfIdfFeaturizer(TfIdfConfig(**model_cfg.get("tfidf", {})))

    X_train = tfidf.fit_transform(train_df["text"])
    y_train = train_df["label"].values

    clf = build_model(model_name, model_cfg, seed)
    clf.fit(X_train, y_train)

    out = {}
    for split_name, df in eval_dfs.items():
        X = tfidf.transform(df["text"])
        y = df["label"].values
        y_pred = clf.predict(X)
        out[split_name] = classification_metrics(y, y_pred)
    return out
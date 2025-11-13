from __future__ import annotations
from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def build_model(name: str, cfg: Dict[str, Any], seed: int = 101):
    if name == "logreg":
        return LogisticRegression(
            C=cfg["C"],
            class_weight=cfg["class_weight"],
            max_iter=2000,
            solver=cfg.get("solver", "liblinear"),
            random_state=seed,
            n_jobs=cfg.get("n_jobs", None),
        )
    if name == "nb":
        return MultinomialNB(alpha=cfg["alpha"])
    if name == "svm":
        return LinearSVC(
            C=cfg["C"],
            class_weight=cfg["class_weight"],
            random_state=seed
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            n_jobs=cfg.get("n_jobs", -1),
            class_weight=cfg.get("class_weight", None),
            random_state=seed
        )

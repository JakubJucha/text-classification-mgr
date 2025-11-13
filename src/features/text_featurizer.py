from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class TfIdfConfig:
    ngram_min: int = 1
    ngram_max: int = 1
    min_df: int = 1
    max_df: float = 1.0
    max_features: int | None = None
    stop_words: str | None = None
    sublinear_tf: bool = False
    binary: bool = False

class TfIdfFeaturizer:
    def __init__(self, cfg: TfIdfConfig):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(cfg.ngram_min, cfg.ngram_max),
            min_df=cfg.min_df,
            max_df=cfg.max_df,
            max_features=cfg.max_features,
            stop_words=cfg.stop_words,
            sublinear_tf=cfg.sublinear_tf,
            binary=cfg.binary,
            strip_accents="unicode",
            lowercase=True,
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)
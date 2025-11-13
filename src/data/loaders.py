import pandas as pd
from dataclasses import dataclass
from typing import  Optional, Literal
from sklearn.model_selection import train_test_split

Split = Literal["train", "validation", "test"]

@dataclass
class LocalCsvCfg:
    files: dict               
    columns: dict              
    val_size: float = 0.1
    test_size: Optional[float] = None  
    stratify: bool = True
    sep: str = ","
    encoding: str = "utf-8"
    random_state: int = 101

def _normalize_text(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    if "text" in columns:
        df["text"] = df[columns["text"]].astype(str)
    elif "text_cols" in columns: 
        df["text"] = df[columns["text_cols"]].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        raise ValueError("columns must specify 'text' or 'text_cols'")
    df["label"] = df[columns["label"]]
    return df[["text","label"]]

def load_local_csv(split: Split, cfg: LocalCsvCfg) -> pd.DataFrame:
   
    train_df_raw = pd.read_csv(cfg.files["train"], sep=cfg.sep, encoding=cfg.encoding)
    train_df = _normalize_text(train_df_raw, cfg.columns)

    test_df = None
    if "test" in cfg.files and cfg.files["test"]:
        test_df_raw = pd.read_csv(cfg.files["test"], sep=cfg.sep, encoding=cfg.encoding)
        test_df = _normalize_text(test_df_raw, cfg.columns)

    y = train_df["label"] if cfg.stratify else None
    if test_df is None and cfg.test_size:
        train_df, test_df = train_test_split(
            train_df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if cfg.stratify else None
        )

    y_train = train_df["label"] if cfg.stratify else None
    train_df, val_df = train_test_split(
        train_df, test_size=cfg.val_size, random_state=cfg.random_state, stratify=y_train if cfg.stratify else None
    )

    out = {"train": train_df.reset_index(drop=True),
           "validation": val_df.reset_index(drop=True),
           "test": (test_df.reset_index(drop=True) if test_df is not None else val_df.reset_index(drop=True))}
    return out[split]
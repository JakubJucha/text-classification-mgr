
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_PATH / "results" / "rebaseline"   
OUT_DIR = BASE_PATH / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_RUNS_CSV  = OUT_DIR / "phase3_runs.csv"
OUT_SUMM_CSV  = OUT_DIR / "phase3_summary.csv"
OUT_RUNS_XLSX = OUT_DIR / "phase3_runs.xlsx"
OUT_SUMM_XLSX = OUT_DIR / "phase3_summary.xlsx"

_METRIC_KEYS = [
    "accuracy", "f1_macro", "precision_macro", "recall_macro",
    "elapsed_sec",
]

_META_KEYS = [
    "phase","dataset","model","seed","timestamp",
]

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _autosize_xlsx(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame):
    ws = writer.sheets[sheet_name]
    for i, col in enumerate(df.columns, 1):
        max_len = max(len(str(col)), *(len(str(v)) for v in df[col].head(500))) + 2
        ws.set_column(i-1, i-1, min(60, max_len))

def load_runs_df() -> pd.DataFrame:
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skip {p.name}: {e}", file=sys.stderr)
            continue

        meta = obj.get("meta") or {}
        metrics = obj.get("metrics") or {}  

        row = {"file": str(p.relative_to(BASE_PATH))}

        for k in _META_KEYS:
            row[k] = meta.get(k)

        for k in _METRIC_KEYS:
            row[k] = metrics.get(k)

        row["model_cfg_json"]     = json.dumps(meta.get("model_cfg", {}), ensure_ascii=False)
        row["dataset_files_json"] = json.dumps(meta.get("files", {}), ensure_ascii=False)

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    num_cols = ["seed"] + _METRIC_KEYS
    df = _coerce_numeric(df, num_cols)
    if "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["dataset","model","timestamp_parsed","file"]).drop(columns=["timestamp_parsed"])

 
    lead = ["phase","dataset","model","seed","timestamp"]
    metrics = [c for c in _METRIC_KEYS if c in df.columns]
    extras = ["file","model_cfg_json","dataset_files_json"]
    cols = [c for c in lead if c in df.columns] + metrics + [c for c in extras if c in df.columns]
    return df[cols]

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grp_keys = [c for c in ["phase","dataset","model"] if c in df.columns]
    agg = (
        df.groupby(grp_keys, dropna=False)
          .agg(
              n_runs=("file","count"),
              seeds=("seed", lambda s: ",".join(sorted({str(x) for x in s if pd.notna(x)}))),
              f1_mean=("f1_macro","mean"),
              f1_std =("f1_macro","std"),
              acc_mean=("accuracy","mean"),
              acc_std =("accuracy","std"),
              prec_mean=("precision_macro","mean"),
              rec_mean =("recall_macro","mean"),
              time_mean=("elapsed_sec","mean"),
              time_sum =("elapsed_sec","sum"),
          )
          .reset_index()
    )
    for col in agg.columns:
        if agg[col].dtype.kind in "fc":
            agg[col] = agg[col].astype(float).round(6)
    return agg

def _save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig", sep=";", na_rep="")

def _save_xlsx(df: pd.DataFrame, path: Path, sheet_name="data"):
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name=sheet_name)
        _autosize_xlsx(xw, sheet_name, df)

def main():
    df = load_runs_df()

    _save_csv(df, OUT_RUNS_CSV)
    _save_xlsx(df, OUT_RUNS_XLSX)
    print(f"Zapisano: {OUT_RUNS_CSV}")
    print(f"Zapisano: {OUT_RUNS_XLSX}")

    summ = summarize(df)
    _save_csv(summ, OUT_SUMM_CSV)
    _save_xlsx(summ, OUT_SUMM_XLSX)
    print(f"Zapisano: {OUT_SUMM_CSV}")
    print(f"Zapisano: {OUT_SUMM_XLSX}")

if __name__ == "__main__":
    main()

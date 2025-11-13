
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
IN_DIR = BASE / "results" / "learning_curve"
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_CSV  = OUT_DIR / "phase5_runs.csv"
RUNS_XLSX = OUT_DIR / "phase5_runs.xlsx"
SUM_CSV   = OUT_DIR / "phase5_summary.csv"
SUM_XLSX  = OUT_DIR / "phase5_summary.xlsx"

def _autosize(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame):
    ws = writer.sheets[sheet]
    for i, col in enumerate(df.columns, 1):
        width = min(60, max(len(str(col)), *(len(str(v)) for v in df[col].head(500))) + 2)
        ws.set_column(i-1, i-1, width)

def load_runs() -> pd.DataFrame:
    rows = []
    for p in sorted(IN_DIR.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skip {p.name}: {e}", file=sys.stderr)
            continue

        meta = obj.get("meta") or {}
        met  = obj.get("metrics") or {}       
        val  = obj.get("val_metrics") or {}    

        row = {

            "phase": meta.get("phase"),
            "timestamp": meta.get("timestamp"),
            "dataset": meta.get("dataset"),
            "model": meta.get("model"),
            "seed": meta.get("seed"),
            "train_frac": meta.get("train_frac"),
            "train_size_used": meta.get("train_size_used"),
            "val_size_used": meta.get("val_size_used"),
            "test_size_used": meta.get("test_size_used"),
       
            "test_accuracy": met.get("accuracy"),
            "test_f1_macro": met.get("f1_macro"),
            "test_precision_macro": met.get("precision_macro"),
            "test_recall_macro": met.get("recall_macro"),
            "test_elapsed_sec": met.get("elapsed_sec"),
       
            "val_accuracy": val.get("accuracy"),
            "val_f1_macro": val.get("f1_macro"),
            "val_precision_macro": val.get("precision_macro"),
            "val_recall_macro": val.get("recall_macro"),
       
            "model_cfg_json": json.dumps(meta.get("model_cfg", {}), ensure_ascii=False),
            "file": str(p.relative_to(BASE)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df


    num_cols = [
        "seed", "train_frac", "train_size_used", "val_size_used", "test_size_used",
        "test_accuracy","test_f1_macro","test_precision_macro","test_recall_macro","test_elapsed_sec",
        "val_accuracy","val_f1_macro","val_precision_macro","val_recall_macro",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

 
    if "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["dataset","model","train_frac","seed","timestamp_parsed"]).drop(columns=["timestamp_parsed"])

  
    cols = [
        "phase","dataset","model","seed","train_frac","train_size_used","val_size_used","test_size_used","timestamp",
        "test_accuracy","test_f1_macro","test_precision_macro","test_recall_macro","test_elapsed_sec",
        "val_accuracy","val_f1_macro","val_precision_macro","val_recall_macro",
        "model_cfg_json","file",
    ]
    return df[cols]

def save_runs(df: pd.DataFrame):
    df.to_csv(RUNS_CSV, index=False, encoding="utf-8-sig", sep=";", na_rep="")
    with pd.ExcelWriter(RUNS_XLSX, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="runs")
        _autosize(xw, "runs", df)

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grp = (
        df.groupby(["dataset","model","train_frac"], dropna=False)
          .agg(
              n_runs=("file","count"),
              seeds=("seed", lambda s: ",".join(sorted({str(x) for x in s if pd.notna(x)}))),
              train_size_used_mean=("train_size_used","mean"),
           
              f1_mean=("test_f1_macro","mean"),
              f1_std =("test_f1_macro","std"),
              acc_mean=("test_accuracy","mean"),
              acc_std =("test_accuracy","std"),
              prec_mean=("test_precision_macro","mean"),
              rec_mean =("test_recall_macro","mean"),
              time_mean=("test_elapsed_sec","mean"),
              time_sum =("test_elapsed_sec","sum"),
             
              val_f1_mean=("val_f1_macro","mean"),
              val_f1_std =("val_f1_macro","std"),
          )
          .reset_index()
    )
    for c in grp.columns:
        if pd.api.types.is_float_dtype(grp[c]):
            grp[c] = grp[c].astype(float).round(6)
    grp = grp.sort_values(["dataset","model","train_frac"]).reset_index(drop=True)
    return grp

def save_summary(df_sum: pd.DataFrame):
    df_sum.to_csv(SUM_CSV, index=False, encoding="utf-8-sig", sep=";", na_rep="")
    with pd.ExcelWriter(SUM_XLSX, engine="xlsxwriter") as xw:
        df_sum.to_excel(xw, index=False, sheet_name="summary")
        _autosize(xw, "summary", df_sum)

def main():
    df = load_runs()
    save_runs(df)
    print(f"Zapisano: {RUNS_CSV}\nZapisano: {RUNS_XLSX}")

    summary = make_summary(df)
    save_summary(summary)
    print(f"Zapisano: {SUM_CSV}\nZapisano: {SUM_XLSX}")

if __name__ == "__main__":
    main()

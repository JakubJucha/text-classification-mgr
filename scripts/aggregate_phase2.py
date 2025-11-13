from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
IN_DIR = BASE / "results" / "tuning"          
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_CSV   = OUT_DIR / "phase2_runs.csv"
RUNS_XLSX  = OUT_DIR / "phase2_runs.xlsx"
SUM_CSV    = OUT_DIR / "phase2_summary.csv"
SUM_XLSX   = OUT_DIR / "phase2_summary.xlsx"
BEST_JSON  = OUT_DIR / "phase2_best_global_configs.json"
BEST_CSV   = OUT_DIR / "phase2_best_global_configs.csv"
BEST_XLSX  = OUT_DIR / "phase2_best_global_configs.xlsx"

TEST_KEYS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "elapsed_sec"]
VAL_KEYS  = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

META_KEYS = [
    "phase","timestamp","dataset","model","seed","combo_id",
]

def _coerce_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _autosize(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame):
    ws = writer.sheets[sheet]
    for i, col in enumerate(df.columns, 1):
        width = min(60, max(len(str(col)), *(len(str(v)) for v in df[col].head(500))) + 2)
        ws.set_column(i-1, i-1, width)

def load_runs_df() -> pd.DataFrame:
    rows = []
    for p in sorted(IN_DIR.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skip {p.name}: {e}", file=sys.stderr)
            continue

        meta = obj.get("meta") or {}
        test = obj.get("metrics") or {}        
        val  = obj.get("val_metrics") or {}    

        row = {
            "file": str(p.relative_to(BASE)),
            "model_cfg_json": json.dumps(meta.get("model_cfg", {}), ensure_ascii=False),
            "files_json": json.dumps(meta.get("files", {}), ensure_ascii=False),
            "columns_json": json.dumps(meta.get("columns", {}), ensure_ascii=False),
        }
       
        for k in META_KEYS:
            row[k] = meta.get(k)
       
        for k in TEST_KEYS:
            row[f"test_{k}"] = test.get(k)
      
        for k in VAL_KEYS:
            row[f"val_{k}"] = val.get(k)

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df


    num_cols = ["seed","combo_id"] + [c for c in df.columns if c.startswith(("test_","val_"))]
    df = _coerce_num(df, num_cols)
    if "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["model","dataset","combo_id","seed","timestamp_parsed"]).drop(columns=["timestamp_parsed"])


    lead = ["phase","model","dataset","seed","combo_id","timestamp"]
    metrics = [c for c in df.columns if c.startswith(("val_","test_"))]
    extras = ["file","model_cfg_json","files_json","columns_json"]
    cols = [c for c in lead if c in df.columns] + metrics + extras
    return df[cols]

def save_runs(df: pd.DataFrame):

    df.to_csv(RUNS_CSV, index=False, encoding="utf-8-sig", sep=";", na_rep="")

    with pd.ExcelWriter(RUNS_XLSX, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="runs")
        _autosize(xw, "runs", df)

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def uniq_join(series):
        vals = [str(x) for x in series.dropna().unique().tolist()]
        return ",".join(sorted(vals))

    grp = df.groupby(["model","combo_id"], as_index=False).agg(
        n_runs          = ("file","count"),
        datasets        = ("dataset", uniq_join),
        seeds           = ("seed",    uniq_join),

        val_f1_mean     = ("val_f1_macro","mean"),
        val_f1_std      = ("val_f1_macro","std"),
        val_acc_mean    = ("val_accuracy","mean"),
        val_acc_std     = ("val_accuracy","std"),
        val_prec_mean   = ("val_precision_macro","mean"),
        val_rec_mean    = ("val_recall_macro","mean"),

        test_f1_mean    = ("test_f1_macro","mean"),
        test_f1_std     = ("test_f1_macro","std"),
        test_acc_mean   = ("test_accuracy","mean"),
        test_acc_std    = ("test_accuracy","std"),
        test_prec_mean  = ("test_precision_macro","mean"),
        test_rec_mean   = ("test_recall_macro","mean"),
        test_time_mean  = ("test_elapsed_sec","mean"),
        test_time_sum   = ("test_elapsed_sec","sum"),

        model_cfg_json  = ("model_cfg_json","first"),
    )


    for c in grp.columns:
        if pd.api.types.is_float_dtype(grp[c]):
            grp[c] = grp[c].astype(float).round(6)

    grp = grp.sort_values(["model","val_f1_mean"], ascending=[True, False]).reset_index(drop=True)
    return grp

def save_summary(df_sum: pd.DataFrame):
    df_sum.to_csv(SUM_CSV, index=False, encoding="utf-8-sig", sep=";", na_rep="")
    with pd.ExcelWriter(SUM_XLSX, engine="xlsxwriter") as xw:
        df_sum.to_excel(xw, index=False, sheet_name="summary")
        _autosize(xw, "summary", df_sum)

def export_best(df_sum: pd.DataFrame):

    if df_sum.empty:
        pd.DataFrame().to_csv(BEST_CSV, index=False, sep=";", encoding="utf-8-sig")
        with open(BEST_JSON, "w", encoding="utf-8") as f:
            f.write("[]")
        return

    best = (df_sum.sort_values("val_f1_mean", ascending=False)
                  .groupby("model", as_index=False)
                  .head(1)
                  .reset_index(drop=True))

    best.to_csv(BEST_CSV, index=False, sep=";", encoding="utf-8-sig")
    with pd.ExcelWriter(BEST_XLSX, engine="xlsxwriter") as xw:
        best.to_excel(xw, index=False, sheet_name="best")
        _autosize(xw, "best", best)

    out = []
    for _, r in best.iterrows():
        out.append({
            "model": r["model"],
            "combo_id": int(r["combo_id"]),
            "val_f1_mean": float(r["val_f1_mean"]) if pd.notna(r["val_f1_mean"]) else None,
            "test_f1_mean": float(r["test_f1_mean"]) if pd.notna(r["test_f1_mean"]) else None,
            "n_runs": int(r["n_runs"]),
            "datasets": r["datasets"],
            "seeds":    r["seeds"],
            "model_cfg": json.loads(r["model_cfg_json"]),
        })
    Path(BEST_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    runs = load_runs_df()
    save_runs(runs)
    print(f"Zapisano: {RUNS_CSV}\nZapisano: {RUNS_XLSX}")

    summary = make_summary(runs)
    save_summary(summary)
    print(f"Zapisano: {SUM_CSV}\nZapisano: {SUM_XLSX}")

    export_best(summary)
    print(f"Zapisano: {BEST_CSV}\nZapisano: {BEST_XLSX}\nZapisano: {BEST_JSON}")

if __name__ == "__main__":
    main()

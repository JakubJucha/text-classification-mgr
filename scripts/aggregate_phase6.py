from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
IN_DIR = BASE / "results" / "cost_profile"
OUT_DIR = BASE / "results"; OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_CSV  = OUT_DIR / "phase6_cost_runs.csv"
RUNS_XLSX = OUT_DIR / "phase6_cost_runs.xlsx"
SUM_CSV   = OUT_DIR / "phase6_cost_summary.csv"
SUM_XLSX  = OUT_DIR / "phase6_cost_summary.xlsx"


def _autosize(xw: pd.ExcelWriter, sheet: str, df: pd.DataFrame):
    ws = xw.sheets[sheet]
    for i, col in enumerate(df.columns, 1):
        width = min(60, max(len(str(col)), *(len(str(v)) for v in df[col].head(500))) + 2)
        ws.set_column(i-1, i-1, width)


def load_runs() -> pd.DataFrame:
    rows = []
    for p in sorted(IN_DIR.glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        meta, met, cst = obj["meta"], obj["metrics"], obj["costs"]
        rows.append({
            "dataset": meta["dataset"],
            "model": meta["model"],
            "seed": meta["seed"],
            "timestamp": meta["timestamp"],
            "n_train": meta["n_train"],
            "n_test": meta["n_test"],
            "feat_dim": meta["feat_dim"],

            "test_accuracy": met["accuracy"],
            "test_f1_macro": met["f1_macro"],
            "test_precision_macro": met["precision_macro"],
            "test_recall_macro": met["recall_macro"],

            "time_fit_tfidf_sec": cst["time_fit_tfidf_sec"],
            "time_fit_clf_sec": cst["time_fit_clf_sec"],
            "time_predict_sec": cst["time_predict_sec"],
            "time_total_train_sec": cst["time_total_train_sec"],
            "time_total_infer_sec": cst["time_total_infer_sec"],
            "model_size_bytes": cst["model_size_bytes"],

            "file": str(p.relative_to(BASE)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    num_cols = [c for c in df.columns if c not in ("dataset","model","timestamp","file")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(
            ["dataset","model","seed","timestamp_parsed"]
        ).drop(columns=["timestamp_parsed"])

    cols = [
        "dataset","model","seed","n_train","n_test","feat_dim","timestamp",
        "test_accuracy","test_f1_macro","test_precision_macro","test_recall_macro",
        "time_fit_tfidf_sec","time_fit_clf_sec","time_predict_sec",
        "time_total_train_sec","time_total_infer_sec",
        "model_size_bytes",
        "file",
    ]
    return df[cols]


def save_runs(df: pd.DataFrame):
    # CSV
    df.to_csv(RUNS_CSV, index=False, encoding="utf-8-sig", sep=";", na_rep="")
    # XLSX
    with pd.ExcelWriter(RUNS_XLSX, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="runs")
        _autosize(xw, "runs", df)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    grp = (
        df.groupby(["dataset","model"], dropna=False)
        .agg(
            n_runs=("file","count"),
            seeds=("seed", lambda s: ",".join(sorted({str(x) for x in s}))),

            f1_mean=("test_f1_macro","mean"),
            f1_std=("test_f1_macro","std"),
            acc_mean=("test_accuracy","mean"),

            train_time_mean=("time_total_train_sec","mean"),
            train_time_median=("time_total_train_sec","median"),
            infer_time_mean=("time_total_infer_sec","mean"),
            infer_time_median=("time_total_infer_sec","median"),

            fit_tfidf_mean=("time_fit_tfidf_sec","mean"),
            fit_clf_mean=("time_fit_clf_sec","mean"),
            pred_time_mean=("time_predict_sec","mean"),

            model_size_mean=("model_size_bytes","mean"),
            feat_dim_mean=("feat_dim","mean"),
        )
        .reset_index()
    )

    for c in grp.columns:
        if pd.api.types.is_float_dtype(grp[c]):
            grp[c] = grp[c].astype(float).round(6)

    return grp.sort_values(["dataset","model"]).reset_index(drop=True)


def save_summary(df_sum: pd.DataFrame):
    df_sum.to_csv(SUM_CSV, index=False, encoding="utf-8-sig", sep=";", na_rep="")

    with pd.ExcelWriter(SUM_XLSX, engine="xlsxwriter") as xw:
        df_sum.to_excel(xw, index=False, sheet_name="summary")
        _autosize(xw, "summary", df_sum)


def main():
    df = load_runs()
    save_runs(df)
    print(f"Zapisano: {RUNS_CSV}\nZapisano: {RUNS_XLSX}")

    sm = make_summary(df)
    save_summary(sm)
    print(f"Zapisano: {SUM_CSV}\nZapisano: {SUM_XLSX}")


if __name__ == "__main__":
    main()

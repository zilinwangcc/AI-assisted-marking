from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import hashlib

# Inputs (ground truth + totals)
TEACHER_JSONL = "data/SAS-Bench/math_test_teacher.jsonl"
TOTALS_FILE = "data/SAS-Bench/cleaned_Math_ShortAns.jsonl"

# list of experiment configs to summarize.
EXPERIMENTS = [
    # rubrics
    {
        "csvs": [
            "results/clean/o3mini_reviewed_v0_floor_mean.csv",
            "results/clean/o3mini_reviewed_v1_floor_mean.csv",
            "results/clean/o3mini_reviewed_v2_floor_mean.csv",
            "results/clean/o3mini_reviewed_v3_floor_mean.csv",
            "results/clean/o3mini_reviewed_v4_floor_mean.csv",
        ],
        "labels": ["v0_FM", "v1_FM", "v2_FM", "v3_FM", "v4_FM"],
        "out_csv": "results/clean/o3mini_rubrics_summary.csv",
    },

    # few-shot vs no-shot vs one-shot
    {
        "csvs": [
            "results/clean/o3mini_vanilla_reviewed_v4_floor_mean.csv",
            "results/clean/o3mini_reviewed_v4_floor_mean.csv",
            "results/clean/o3mini_few_shot_reviewed_v4_floor_mean.csv",
        ],
        "labels": ["o3mini_no_shot_FM", "o3mini_one_shot_FM", "o3mini_few_shot_FM"],
        "out_csv": "results/clean/o3mini_shots_ablation_summary.csv",
    },

    # strict, lenient, neutral
    {
        "csvs": [
            "results/clean/o3mini_reviewed_v4_floor_mean.csv",
            "results/clean/o3mini_lenient_reviewed_v4_floor_mean.csv",
            "results/clean/o3mini_netural_reviewed_v4_floor_mean.csv",
        ],
        "labels": ["o3mini_strict_FM", "o3mini_lenient_FM", "o3mini_neutral_FM"],
        "out_csv": "results/clean/o3mini_persona_evaluation_summary.csv",
    },

    # with/without analysis & reference
    {
        "csvs": [
            "results/clean/o3mini_noanalysis_reviewed_v4_floor_mean.csv",
            "results/clean/o3mini_no_reference_reviewed_v4_floor_mean.csv",
            "results/clean/o3mini_reviewed_v4_floor_mean.csv",
        ],
        "labels": ["o3mini_no_analysis_FM", 
                   "o3mini_no_reference_FM", 
                   "o3mini_with_analysis_reference_FM"],
        "out_csv": "results/clean/o3mini_content_evaluation_summary.csv",
    },

    # CoT vs no-CoT
    {
        "csvs": [
            "results/clean/o3mini_no_CoT_reviewed_v4_floor_mean.csv",
            "results/clean/o3mini_reviewed_v4_floor_mean.csv",
        ],
        "labels": ["o3mini_no_CoT_FM", "o3mini_with_CoT_FM"],
        "out_csv": "results/clean/o3mini_CoT_summary.csv",
    },

    # model comparison
    {
        "csvs": [
            "results/clean/o3mini_reviewed_v4_floor_mean.csv",
            "results/clean/o3_reviewed_v4_floor_mean.csv",
            "results/clean/gpt-4o_reviewed_v4_floor_mean.csv",
        ],
        "labels": ["o3mini", "o3", "gpt-4o"],
        "out_csv": "results/clean/model_summary.csv",
    },

    # ensemble
        {
        "csvs": [
            "results/predictions/o3_grader_20250904_114835_fbb8be.csv",
            "results/predictions/o3_grader_20250904_124104_7b7e6d.csv",
            "results/predictions/o3_grader_20250904_133336_d613f8.csv",
            "results/predictions/o3_grader_20250904_142630_896d42.csv",
            "results/predictions/o3_grader_20250904_151418_f0b5eb.csv",
            "results/clean/o3_reviewed_v4_mode.csv",
            "results/clean/o3_reviewed_v4_floor_mean.csv",

        ],
        "labels": ["First_attempt", 
                   "Second_attempt", 
                   "Third_attempt", 
                   "Fourth_attempt",
                   "Fifth_attempt", 
                   "o3_mode", 
                   "o3_floor_mean"],
        "out_csv": "results/clean/ensemble_summary.csv",
    },
]


def load_preds(pred_csv: str | Path) -> pd.DataFrame:
    """
    Final aggregated predictions CSV.
    They contain: script_id, model_score.
    """
    df = pd.read_csv(pred_csv)
    if "script_id" not in df.columns or "model_score" not in df.columns:
        raise ValueError(f"{pred_csv} must have columns: script_id, model_score")
    for col in ["model_score", "std_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["script_id"] = df["script_id"].astype(str)
    return df


def load_teacher(teacher_jsonl: str | Path) -> pd.DataFrame:
    """
    Load teacher awarded marks from JSONL
    """
    rows = []
    with open(teacher_jsonl, "r", encoding="utf8") as f:
        for line in f:
            o = json.loads(line)
            rows.append({"script_id": str(o["id"]), "teacher_score": o["awarded_marks"]})
    df = pd.DataFrame(rows)
    df["teacher_score"] = pd.to_numeric(df["teacher_score"], errors="coerce")
    return df


def load_teacher_with_qid(teacher_jsonl: str | Path) -> pd.DataFrame:
    """
    Variant that also keeps question_id.
    """
    rows = []
    with open(teacher_jsonl, "r", encoding="utf8") as f:
        for line in f:
            o = json.loads(line)
            rows.append({
                "script_id": str(o["id"]),
                "teacher_score": o["awarded_marks"],
                "question_id": o["question_id"]
            })
    df = pd.DataFrame(rows)
    df["teacher_score"] = pd.to_numeric(df["teacher_score"], errors="coerce")
    return df


def load_totals(file: str | Path) -> pd.DataFrame:
    """
    File with per-script max marks， expected column name: total.
    """
    label = ["total"]
    p = Path(file)
    
    rows = []
    with open(p, "r", encoding="utf8") as f:
        for line in f:
            o = json.loads(line)
            sid = str(o.get("script_id", o.get("id")))
            tot = None
            for k in label:
                if k in o:
                    tot = o[k]
                    break
            rows.append({"script_id": sid, "total": tot})
    df = pd.DataFrame(rows)

    df["script_id"] = df["script_id"].astype(str)
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    return df[["script_id", "total"]]


# Metrics

def _avg_run_std(df_preds: pd.DataFrame) -> float:
    """
    Return average per-script std across runs.
    - If a 'std_score' column exists, average it.
    - Else if model_score_0.. exist, compute per-row std and average.
    - Else return NaN.
    """
    if "std_score" in df_preds.columns:
        return pd.to_numeric(df_preds["std_score"], errors="coerce").mean()

    run_cols = [c for c in df_preds.columns if c.startswith("model_score_")]
    if run_cols:
        tmp = df_preds[run_cols].apply(pd.to_numeric, errors="coerce")
        return tmp.std(axis=1, ddof=1).mean()
    return float("nan")


def _per_file_metrics(merged: pd.DataFrame, avg_run_std: float) -> dict:
    """
    merged must have: model_score, teacher_score, total (>0)
    """
    d = merged.dropna(subset=["model_score", "teacher_score", "total"]).copy()
    d = d[d["total"] > 0]

    # percentages for QWK
    y_true = np.round(100 * d["teacher_score"] / d["total"]).astype(int)
    y_pred = np.round(100 * d["model_score"]   / d["total"]).astype(int)

    # errors
    err = d["model_score"] - d["teacher_score"]
    tot = d["total"]

    # points
    mae_points = err.abs().mean()
    mae_pct = 100 * (err.abs() / tot).mean()

    # percents
    signed_bias    = (100 * err / tot).mean() 
    mean_abs_gap_pct  = (100 * err.abs() / tot).mean()
    weighted_bias_pct = 100 * err.sum() / tot.sum()
    wape_pct          = 100 * err.abs().sum() / tot.sum()

    # qwk
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    over_rate  = (err > 0).mean()
    under_rate = (err < 0).mean()
    equal_rate  = (err == 0).mean() 

    return {
        "n": len(d),
        "QWK": qwk,
        "MAE_points": mae_points,
        "MAE_pct": mae_pct,
        "signed_bias": signed_bias,
        "weighted_bias_pct": weighted_bias_pct,
        "mean_abs_gap_pct": mean_abs_gap_pct,
        "WAPE_pct": wape_pct,
        "over_rate": over_rate,
        "under_rate": under_rate,
        "equal_rate": equal_rate,
        "avg_std_over_runs": avg_run_std,
    }


def evaluate_one(pred_csv: str | Path,
                 teacher_jsonl: str | Path,
                 totals_file: str | Path,
                 label: str | None = None) -> dict:
    """
    Evaluate one final CSV and return a metrics dict (with 'source' label).
    """
    preds  = load_preds(pred_csv)
    teach  = load_teacher(teacher_jsonl)
    totals = load_totals(totals_file)

    avg_std = _avg_run_std(preds)

    df = preds.merge(teach, on="script_id", how="inner", validate="1:1")
    df = df.merge(totals, on="script_id", how="left")

    # keep only rows with valid totals
    df = df.dropna(subset=["total"])
    df = df[df["total"] > 0].copy()

    m = _per_file_metrics(df, avg_std)
    m["source"] = label if label is not None else Path(pred_csv).stem
    return m


def evaluate_many(pred_csvs: list[str | Path],
                  teacher_jsonl: str | Path,
                  totals_file: str | Path,
                  labels: list[str] | None,
                  out_csv: str | Path) -> pd.DataFrame:
    """
    Evaluate multiple final CSVs and write a summary table.
    Returns the summary df.
    """
    if labels is None:
        labels = [Path(p).stem for p in pred_csvs]
    if len(labels) != len(pred_csvs):
        raise ValueError("labels length must match pred_csvs length")

    rows = []
    for lab, path in zip(labels, pred_csvs):
        rows.append(evaluate_one(path, teacher_jsonl, totals_file, label=lab))

    summary = pd.DataFrame(rows, columns=[
        "source", "n", "QWK", "MAE_points", "MAE_pct",
        "signed_bias", "weighted_bias_pct",
        "mean_abs_gap_pct", "WAPE_pct",
        "over_rate", "under_rate", "equal_rate",
        "avg_std_over_runs",
    ])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"Wrote summary to {out_csv}")
    return summary


def merge_for_examples(pred_csv: str | Path,
                       teacher_jsonl: str | Path,
                       totals_file: str | Path) -> pd.DataFrame:
    """
    Return merged df with script_id, teacher_score, model_score, total.
    """
    preds  = load_preds(pred_csv)
    teach  = load_teacher(teacher_jsonl)
    totals = load_totals(totals_file)

    df = preds.merge(teach, on="script_id", how="inner", validate="1:1")
    df = df.merge(totals, on="script_id", how="left")
    df = df.dropna(subset=["model_score", "teacher_score", "total"]).copy()
    df = df[df["total"] > 0]

    df["model_score"]   = pd.to_numeric(df["model_score"], errors="coerce")
    df["teacher_score"] = pd.to_numeric(df["teacher_score"], errors="coerce")
    df["total"]         = pd.to_numeric(df["total"], errors="coerce")
    return df[["script_id", "teacher_score", "model_score", "total"]]


# print first K over/under-marked examples
def print_top_over_under_samples(pred_csv: str | Path,
                                 teacher_jsonl: str | Path,
                                 totals_file: str | Path,
                                 k: int = 10,
                                 label: str | None = None) -> None:
    """
    print the first K over-marked and under-marked scripts:
      script_id, teacher_score, model_score, total
    """
    df = merge_for_examples(pred_csv, teacher_jsonl, totals_file)
    df = df.copy()
    df["gap_points"] = df["model_score"] - df["teacher_score"]

    over  = df[df["gap_points"] > 0].sort_values("gap_points", ascending=False).head(k)
    under = df[df["gap_points"] < 0].sort_values("gap_points", ascending=True).head(k)

    tag = f"[{label}]" if label else f"[{Path(pred_csv).stem}]"

    print(f"\nTop {len(over)} OVER-marked {tag}:")
    print(over[["script_id", "teacher_score", "model_score", "total"]]
          .to_string(index=False, float_format=lambda x: f"{x:.0f}"))

    print(f"\nTop {len(under)} UNDER-marked {tag}:")
    print(under[["script_id", "teacher_score", "model_score", "total"]]
          .to_string(index=False, float_format=lambda x: f"{x:.0f}"))


def _stable_item_id(question_text: str) -> str:
    """
    group all student answers for the same together together
    """
    h = hashlib.md5((question_text or "").encode("utf-8")).hexdigest()
    return f"item_{h[:8]}"



if __name__ == "__main__":
    # rubrics
    csvs = [
        "results/clean/o3mini_reviewed_v0_floor_mean.csv",
        "results/clean/o3mini_reviewed_v1_floor_mean.csv",
        "results/clean/o3mini_reviewed_v2_floor_mean.csv",
        "results/clean/o3mini_reviewed_v3_floor_mean.csv",
        "results/clean/o3mini_reviewed_v4_floor_mean.csv",
    ]
    labels = ["v0_FM", "v1_FM", "v2_FM", "v3_FM", "v4_FM"]

    # few-shot vs no-shot vs one-shot
    csvs = [
        "results/clean/o3mini_vanilla_reviewed_v4_floor_mean.csv",
        "results/clean/o3mini_reviewed_v4_floor_mean.csv",
        "results/clean/o3mini_few_shot_reviewed_v4_floor_mean.csv",
    ]
    labels = ["o3mini_no_shot_FM", "o3mini_one_shot_FM", "o3mini_few_shot_FM"]

    # strict, lenient, neutral
    csvs = [
        "results/clean/o3mini_reviewed_v4_floor_mean.csv",
        "results/clean/o3mini_lenient_reviewed_v4_floor_mean.csv",
        "results/clean/o3mini_netural_reviewed_v4_floor_mean.csv",
    ]
    labels = ["o3mini_strict_FM", "o3mini_lenient_FM", "o3mini_neutral_FM"]

    # o3mini with/without analysis, and reference
    csvs = [
        "results/clean/o3mini_no_analysis_reviewed_v4_floor_mean.csv",
        "results/clean/o3mini_no_reference_reviewed_v4_floor_mean.csv",
        "results/clean/o3mini_reviewed_v4_floor_mean.csv",
    ]
    labels = ["o3mini_no_analysis_FM", "o3mini_no_reference_FM", "o3mini_with_analysis_reference_FM"]

    # CoT vs no-CoT
    csvs = [
        "results/clean/o3mini_no_CoT_reviewed_v4_floor_mean.csv",
        "results/clean/o3mini_reviewed_v4_floor_mean.csv",
    ]
    labels = ["o3mini_no_CoT_FM", "o3mini_with_CoT_FM"]
    
    # model
    csvs = [
        "results/clean/o3mini_reviewed_v4_floor_mean.csv",
        "results/clean/o3_reviewed_v4_floor_mean.csv",
        "results/clean/gpt-4o_reviewed_v4_floor_mean.csv",
    ]
    labels = ["o3mini", "o3", "gpt-4o"]

    # evaluate each experiment
    for exp in EXPERIMENTS:
        evaluate_many(exp["csvs"], TEACHER_JSONL,
            TOTALS_FILE, labels=exp["labels"], out_csv=exp["out_csv"],
        )
from pathlib import Path
import pandas as pd
import numpy as np
import os


# Ensemble size
N = 5
AVG_PER_CALL = True

# Candidate column names in the token CSV for the per-script token value
TOKEN_COL_CANDIDATES = ["avg_tokens", 'tokens_used']

QWK_LABEL_COL = "source"
QWK_VALUE_COL = "QWK"

# Experiments
token_csv_groups = [
    # rubric type experiment Exp A
    ["results/clean/o3mini_reviewed_v0_floor_mean.csv",
     "results/clean/o3mini_reviewed_v1_floor_mean.csv",
     "results/clean/o3mini_reviewed_v2_floor_mean.csv",
     "results/clean/o3mini_reviewed_v3_floor_mean.csv",
     "results/clean/o3mini_reviewed_v4_floor_mean.csv",],

    # shots experiment Exp B
    ["results/clean/o3mini_reviewed_v4_floor_mean.csv", # 1-shot, the baseline from Exp A
     "results/clean/o3mini_vanilla_reviewed_v4_floor_mean.csv", # 0-shot
     "results/clean/o3mini_few_shot_reviewed_v4_floor_mean.csv"], # few-shot,

    # model experiment Exp`F`
    ["results/clean/o3mini_reviewed_v4_floor_mean.csv",
     "results/clean/o3_reviewed_v4_floor_mean.csv",
     "results/clean/gpt-4o_reviewed_v4_floor_mean.csv"],

    # ensemble Exp`G`
    ["results/predictions/o3_grader_20250904_114835_fbb8be.csv",
    "results/predictions/o3_grader_20250904_124104_7b7e6d.csv",
    "results/predictions/o3_grader_20250904_133336_d613f8.csv",
    "results/predictions/o3_grader_20250904_142630_896d42.csv",
    "results/predictions/o3_grader_20250904_151418_f0b5eb.csv",
    "results/clean/o3_reviewed_v4_mode.csv",
    "results/clean/o3_reviewed_v4_floor_mean.csv",],
]

qwk_csvs = [
    # Exp A
    "results/clean/o3mini_rubrics_summary.csv",

    # Exp B
    "results/clean/o3mini_shots_ablation_summary.csv",

    # Exp F
    "results/clean/model_summary.csv",

    # Exp G
    "results/clean/ensemble_summary.csv",
]

output_csvs = [
    # Exp A
    "results/clean/icer/o3mini_rubrics_icer.csv",
    # Exp B
    "results/clean/icer/o3mini_shots_icer.csv",
    # Exp F
    "results/clean/icer/model_icer.csv",
    # Exp G
    "results/clean/icer/o3_ensemble_icer.csv",

]


def reorder_rows(df: pd.DataFrame, new_first_index: int) -> pd.DataFrame:
    """
    Move row for the newbaseline to the to
    """
    if len(df) <= new_first_index:
        return df.copy()
    order = [new_first_index] + [i for i in range(len(df)) if i != new_first_index]
    return df.iloc[order].reset_index(drop=True)


def find_col(df: pd.DataFrame, candidates) -> str:
    """
    Returns the original column name that matches the first candidate.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
        

def _norm(s: str) -> str:
    """
    Normalize strings for matching
    """
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")

def tokens_per_ensemble_from_tokens_csv(csv_path: str) -> tuple[float, int, str]:
    """
    Read a per-script token CSV and compute total tokens for an ensemble.
    Returns:
      T_c(float) : total tokens across all scripts for one ensemble (per condition)
      S(int): number of scripts in the CSV
      label(str): filename used as the condition label
    Uses globals AVG_PER_CALL and N to scale per-script cost. As the ensemble expperiment
    has single pass and enemble.
    """
    p = Path(csv_path)
    label = p.stem
    df = pd.read_csv(p)
    col = find_col(df, TOKEN_COL_CANDIDATES)
    vals = df[col].astype(float).values
    if AVG_PER_CALL:
        per_script_ensemble = N * vals # N calls per script
    else:
        per_script_ensemble = vals # single call per script
    T_c = float(per_script_ensemble.sum())
    S = int(len(vals))
    return T_c, S, label

def qwk_map_from_csv(csv_path: str) -> dict:
    """
    Read the summary CSV and build a map: label -> QWK value
    use the QWK_LABEL_COL and QWK_VALUE_COL.
    """
    df = pd.read_csv(csv_path)
    # resolve columns case-insensitively
    label_col = find_col(df, [QWK_LABEL_COL])
    value_col = find_col(df, [QWK_VALUE_COL])
    m = {}
    for _, row in df.dropna(subset=[label_col, value_col]).iterrows():
        m[_norm(str(row[label_col]))] = float(row[value_col])
    return m

# Main

def run_one_experiment(token_csvs: list[str], qwk_csv: str, out_csv: str):
    """
    For a single experiment:
        - Load the QWK map from the summary CSV.
        - For each token CSV, compute tokens_per_ensemble
        - Compute deltas vs the *first* row as baseline.
        - Write the ICER table to out_csv.
    """

    # Load QWK map
    qwk_map = qwk_map_from_csv(qwk_csv)

    rows = []
    for idx, tok_csv in enumerate(token_csvs):

        # Decide whether to use AVG_PER_CALL based on experiment
        is_exp_g = "ensemble" in os.path.basename(out_csv)
        use_avg_per_call = not (is_exp_g and idx < 5)  # First 5 rows = False, last 2 = True
        override_n = 1 if is_exp_g and idx < 5 else N

        # Temporarily override global AVG_PER_CALL
        original_avg = AVG_PER_CALL
        original_n = N
        globals()["AVG_PER_CALL"] = use_avg_per_call
        globals()["N"] = override_n

        T_c, S, label = tokens_per_ensemble_from_tokens_csv(tok_csv)

        # Restore globals
        globals()["AVG_PER_CALL"] = original_avg
        globals()["N"] = original_n

        # Find QWK value
        key = _norm(label)
        Q = qwk_map.get(key, None)
        if Q is None:
            dfq = pd.read_csv(qwk_csv)
            value_col = find_col(dfq, [QWK_VALUE_COL])

            Q = float(dfq[value_col].iloc[idx])
        rows.append({
            "condition_index": idx,
            "condition_label": label,
            "tokens_csv": tok_csv,
            "scripts": S,
            "ensemble_N": override_n,
            "tokens_per_ensemble": T_c,
            "QWK": Q,
        })

    # Baseline = first in the list
    base_T = rows[0]["tokens_per_ensemble"]
    base_Q = rows[0]["QWK"]

    # Compute deltas and "tokens per +1 QWK percentage point"
    for r in rows:
        dT = r["tokens_per_ensemble"] - base_T
        dQ = r["QWK"] - base_Q
        r["delta_tokens"] = dT
        r["delta_QWK"] = dQ
        r["tokens_per_+1_QWKpct"] = (dT / (dQ * 100.0)) if abs(dQ) > 1e-12 else np.nan

    out_df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


def run_all():
    # Iterate over experiment groups
    for token_csvs, qwk_csv, out_csv in zip(token_csv_groups, qwk_csvs, output_csvs):
        # only adjust the shots summary since the baseline is in the second row
        if "shots" in os.path.basename(qwk_csv) or "CoT" in os.path.basename(qwk_csv):
            df_summary = pd.read_csv(qwk_csv)
            df_summary = reorder_rows(df_summary, 1)
            qwk_csv_reordered = os.path.splitext(out_csv)[0] + "__reordered.csv"
            df_summary.to_csv(qwk_csv_reordered, index=False)
            run_one_experiment(token_csvs, qwk_csv_reordered, out_csv)

        else:
            run_one_experiment(token_csvs, qwk_csv, out_csv)

if __name__ == "__main__":
    run_all()

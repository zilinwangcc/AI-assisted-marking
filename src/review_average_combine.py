import pandas as pd
import numpy as np
import subprocess
import os
import glob
import re
from pathlib import Path
from typing import Dict

# ----------------- Config -----------------
NUM_RUNS      = 5
DATASET_PATH  = "data/SAS-Bench/math_test_student.jsonl"
PROMPT_STYLE  = "one_shot"
PERSONA       = "strict"
MODEL         = "o3-mini"  # "o3"  "gpt-4o"  "o3-mini"
TEMP          = "1"
OUTPUT_DIR    = Path("results/clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIRST_ATTEMPT_LABEL = "first_attempt"
PRED_DIR = Path("results/predictions")

# Map rubric versions to rubric paths (v0 uses a different grader, no rubric)

RUBRIC_MAP = {
    0: None,
    1: "rubrics/math_simplified_rubrics_multi.jsonl",  # simplified rubrics
    2: "rubrics/math_refined_simplified_rubrics_multi.jsonl",
    3: "rubrics/math_initial_rubrics_multi.jsonl",  # initial full rubrics
    4: "rubrics/math_refined_rubrics_multi.jsonl",
}

OUT_PREFIX = "o3mini_noanalysis_reviewed_v{v}_{strategy}.csv"


def normalize_for_eval(in_csv: Path, out_csv: Path) -> Path:
    """
    Make a minimal predictions file the evaluator expects:
    just ['script_id', 'model_score'] with the same dtypes.
    """
    df = pd.read_csv(in_csv)
    if "script_id" not in df.columns or "model_score" not in df.columns:
        raise ValueError(f"{in_csv} missing required columns ['script_id','model_score']")
    df_min = df[["script_id", "model_score"]].copy()
    # ensure numeric model_score (some runs might come as strings)
    df_min["model_score"] = pd.to_numeric(df_min["model_score"], errors="raise")
    df_min.to_csv(out_csv, index=False)
    return out_csv


# -------------- Helpers -------------------

def _new_files_after(run_fn, pattern="*.csv"):
    """Run a callable, then return the set of new files in PRED_DIR matching pattern."""
    before = set(map(Path, glob.glob(str(PRED_DIR / pattern))))
    run_fn()
    after = set(map(Path, glob.glob(str(PRED_DIR / pattern))))
    new = sorted(list(after - before), key=os.path.getctime)
    if not new:
        newest = max(after, key=os.path.getctime) if after else None
        return [newest] if newest else []
    return new

_qwk_re   = re.compile(r"(?:Quadratic|QWK)[^\d\-+]*([-+]?\d*\.\d+|\d+)", re.I)
# _cohen_re = re.compile(r"Cohen[’']?s?\s*κ[^\d\-+]*([-+]?\d*\.\d+|\d+)", re.I)
_mae_re   = re.compile(r"Mean\s+Abs(?:olute)?\s+Err\.?\s*:\s*([-+]?\d*\.\d+|\d+)", re.I)

def run_evaluator(csv_path: Path) -> Dict[str, float]:
    """
    Run src.evaluation on csv_path, parse stdout (or stderr) for metrics.
    Raises with full logs if returncode != 0 so you can see the reason.
    """
    proc = subprocess.run(
        ["python", "-m", "src.evaluations.evaluate", str(csv_path)],
        capture_output=True, text=True
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(f"Evaluation failed on {csv_path} (exit {proc.returncode}). Output:\n{out}")

    return {
        "qwk": float(_qwk_re.search(out).group(1)) if _qwk_re.search(out) else np.nan,
        # "cohen_kappa": float(_cohen_re.search(out).group(1)) if _cohen_re.search(out) else np.nan,
        "mae": float(_mae_re.search(out).group(1)) if _mae_re.search(out) else np.nan,
        "n": pd.read_csv(csv_path, usecols=["script_id"]).shape[0],
        # "stdout": out,
    }

def run_one_grade(version: int, rubric_path: str | None):
    """Run NUM_RUNS grading jobs for a given rubric version. Returns list of per-run CSV Paths."""
    print(f"\n=== Rubric version {version} ===")
    per_run_files: list[Path] = []

    for i in range(NUM_RUNS):
        print(f"Run {i+1}/{NUM_RUNS} (v{version})")
        if rubric_path is not None:
            # Standard grader with rubric
            def _call():
                subprocess.run([
                    "python", "-m", "src.grader_SAS",
                    "--dataset", DATASET_PATH,
                    "--rubric", rubric_path,
                    "--persona", PERSONA,
                    "--model", MODEL,
                    "--prompt_style", PROMPT_STYLE,
                    "--temperature", str(TEMP),
                ], check=True)
        else:
            # Special grader for v0 (no rubric)
            def _call():
                subprocess.run([
                    "python", "-m", "src.grader_no_rubric",
                    "--dataset", DATASET_PATH,
                    "--persona", PERSONA,
                    "--model", MODEL,
                    "--prompt_style", PROMPT_STYLE,
                    "--temperature", str(TEMP),
                ], check=True)

        new_files = _new_files_after(_call, pattern="*.csv")
        if not new_files:
            raise RuntimeError("No new prediction CSV detected in results/predictions/")
        per_run_files.append(new_files[-1])  # take the last from this run
        print(f"captured {per_run_files[-1].name}")

    return per_run_files


def aggregate_runs(per_run_files: list[Path], version: int):
    """
    Aggregate runs, write 4 strategy CSVs, and return:
      produced: dict[str, Path]       # strategy -> CSV path
      mean_std_across_runs: float
      base_out: pd.DataFrame          # merged per-script table incl. per-run cols
    """
    dfs = []
    tokens_list = []

    for i, p in enumerate(per_run_files):
        df = pd.read_csv(p)[["script_id", "model_score", "tokens_used"]].copy()
        df = df.rename(columns={"model_score": f"model_score_{i}"})
        dfs.append(df)
        tokens_list.append(df[["script_id", "tokens_used"]].rename(columns={"tokens_used": f"tokens_used_{i}"}))

    base = dfs[0]
    for df in dfs[1:]:
        base = base.merge(df, on="script_id", how="outer")

    # Merge tokens separately
    tokens_base = tokens_list[0]
    for df in tokens_list[1:]:
        tokens_base = tokens_base.merge(df, on="script_id", how="outer")

    token_cols = [c for c in tokens_base.columns if c.startswith("tokens_used_")]
    tokens_base["avg_tokens"] = tokens_base[token_cols].mean(axis=1, skipna=True)
    base = base.merge(tokens_base[["script_id", "avg_tokens"]], on="script_id", how="left")

    mark_cols = [c for c in base.columns if c.startswith("model_score_")]
    base["avg_score"]    = base[mark_cols].mean(axis=1, skipna=True)
    base["std_score"]    = base[mark_cols].std(axis=1, skipna=True)
    base["median_score"] = base[mark_cols].median(axis=1, skipna=True)

    def row_mode(row):
        s = pd.to_numeric(row[mark_cols], errors="coerce").dropna().astype(int)
        return s.mode().iloc[0] if not s.empty else np.nan

    base["mode"] = base.apply(row_mode, axis=1).astype("Int64")

    mean_std_across_runs = float(base["std_score"].mean())

    strategies = {
        "top_mean":   base["avg_score"].apply(np.ceil).astype(int),
        "floor_mean": base["avg_score"].apply(np.floor).astype(int),
        "median":     base["median_score"].astype(int),
        "mode":       base["mode"],
    }

    produced: dict[str, Path] = {}
    for strategy, scores in strategies.items():
        out_df = base[["script_id"]].copy()
        out_df["model_score"] = scores
        for c in mark_cols + ["avg_score", "std_score", "median_score", "avg_tokens"]:
            out_df[c] = base[c]
        out_path = OUTPUT_DIR / OUT_PREFIX.format(v=version, strategy=strategy)
        out_df.to_csv(out_path, index=False)
        produced[strategy] = out_path
        print(f"Saved v{version} {strategy}: {out_path}")

    # also return a wide table with the fused columns attached
    base_out = base.copy()
    for k, col in strategies.items():
        base_out[k] = col

    return produced, mean_std_across_runs, base_out


# Main
def main():
    summary_rows = []
    all_versions_wide = []

    for v, rpath in RUBRIC_MAP.items():
        # 1) run graders
        run_csvs = run_one_grade(version=v, rubric_path=rpath)

        # 1a) evaluate the first attempt (run 1) and add to final CSV
        first_csv = run_csvs[0]
        first_norm = OUTPUT_DIR / f"o3mini_noanalysis_reviewed_v{v}_{FIRST_ATTEMPT_LABEL}.csv"
        first_norm = normalize_for_eval(first_csv, first_norm)

        first_metrics = run_evaluator(first_norm)
        summary_rows.append({
            "rubric_version": v,
            "strategy": FIRST_ATTEMPT_LABEL,
            "qwk": first_metrics["qwk"],
            "mae": first_metrics["mae"],
            "std_across_runs": np.nan,
        })


        # 2) aggregate to strategies
        # produced, mean_std_across_runs = aggregate_runs(run_csvs, version=v)
        produced, mean_std_across_runs, base_df = aggregate_runs(run_csvs, version=v)

        base_df.insert(0, "rubric_version", v)  # keep which version it was
        all_versions_wide.append(base_df)

        # 3) evaluate each strategy CSV and capture metrics
        for strategy, csv_path in produced.items():
            metrics = run_evaluator(csv_path)
            print(f"Eval v{v} {strategy} -> QWK={metrics['qwk']:.3f}, MAE={metrics['mae']:.3f}, N={metrics['n']}")
            summary_rows.append({
                "rubric_version": v,
                "strategy": strategy,
                # "pred_csv": str(csv_path),
                "qwk": metrics["qwk"],
                # "cohen_kappa": metrics["cohen_kappa"],
                "mae": metrics["mae"],
                "std_across_runs": mean_std_across_runs,
            })

if __name__ == "__main__":
    main()

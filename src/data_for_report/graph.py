from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Choose experiments to run
ACTIVE = ["rubrics", "shots", "persona", "content", "cot", "model", "ensemble"] 

# Choose which graphs to make
DO_SUMMARY = True   
DO_LINE    = True

N_RUNS_OVERRIDE = 5

# helpers
def load_predicted_marks_map(jsonl_path: Path) -> dict:
    m = {}
    with jsonl_path.open("r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            m[str(obj["id"])] = int(obj["full_marks"])
    return m

def load_teacher_score_map(jsonl_path: Path) -> dict:
    m = {}
    with jsonl_path.open("r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            ts = obj.get("teacher_score", obj.get("awarded_marks"))
            if ts is not None:
                m[str(obj["id"])] = int(ts)
    return m

def _safe_read_csv(path: Path, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not usecols:
        return df
    keep = [c for c in usecols if c in df.columns]
    return df[keep].copy()


EXPERIMENTS = [
    {
        "exp_name": "rubrics",
        "model_label": "o3-mini_rubrics_experiment",
        "n_runs": 5,
        "summary_csv": Path("results/clean/o3mini_rubrics_summary.csv"),
        "out_dir": Path("results/graphs/o3mini_rubrics"),
        "name_map": [("v0_FM","No rubrics"),("v1_FM","Simplified"),("v2_FM","Simplified-Refined"),
                     ("v3_FM","Full"),("v4_FM","Full-Refined")],
        "csv_files": [("No Rubric", Path("results/clean/o3mini_reviewed_v0_floor_mean.csv")),
                      ("Simplified", Path("results/clean/o3mini_reviewed_v1_floor_mean.csv")),
                      ("Simplified-Refined", Path("results/clean/o3mini_reviewed_v2_floor_mean.csv")),
                      ("Full", Path("results/clean/o3mini_reviewed_v3_floor_mean.csv")),
                      ("Full-Refined", Path("results/clean/o3mini_reviewed_v4_floor_mean.csv"))],
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        "line_title": "Human vs AI Grades (o3-mini, rubrics) with Error Bars (Floor Mean Review)",
    },
    {
        "exp_name": "shots",
        "model_label": "o3-mini_shots_experiment",
        "n_runs": 5,
        "summary_csv": Path("results/clean/o3mini_shots_ablation_summary.csv"),
        "out_dir": Path("results/graphs/o3mini_shots_ablation"),
        "name_map": [("o3mini_no_shot_FM","Zero-shot"),("o3mini_one_shot_FM","One-shot"),("o3mini_few_shot_FM","Few-shot")],
        "csv_files": [("Zero-shot", Path("results/clean/o3mini_vanilla_reviewed_v4_floor_mean.csv")),
                      ("One-shot", Path("results/clean/o3mini_reviewed_v4_floor_mean.csv")),
                      ("Few-shot", Path("results/clean/o3mini_few_shot_reviewed_v4_floor_mean.csv"))],
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
        "line_title": "Human vs AI Grades (o3-mini, shots) with Error Bars (Floor Mean Review)",
    },
    {
        "exp_name": "persona",
        "model_label": "o3-mini_persona_experiment",
        "n_runs": 5,
        "summary_csv": Path("results/clean/o3mini_persona_evaluation_summary.csv"),
        "out_dir": Path("results/graphs/o3mini_persona"),
        "name_map": [("o3mini_neutral_FM","Neutral"),("o3mini_lenient_FM","Lenient"),("o3mini_strict_FM","Strict")],
        "csv_files": [("Lenient", Path("results/clean/o3mini_lenient_reviewed_v4_floor_mean.csv")),
                      ("Neutral", Path("results/clean/o3mini_netural_reviewed_v4_floor_mean.csv")),
                      ("Strict",  Path("results/clean/o3mini_reviewed_v4_floor_mean.csv"))],
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
        "line_title": "Human vs AI Grades (o3-mini, persona) with Error Bars (Floor Mean Review)",
    },
    {
        "exp_name": "content",
        "model_label": "o3-mini_content_experiment",
        "n_runs": 5,
        "summary_csv": Path("results/clean/o3mini_content_evaluation_summary.csv"),
        "out_dir": Path("results/graphs/o3mini_content"),
        "name_map": [("o3mini_no_reference_FM","No reference"),("o3mini_no_analysis_FM","No analysis"),("o3mini_with_analysis_reference_FM","Original")],
        "csv_files": [("No reference", Path("results/clean/o3mini_no_reference_reviewed_v4_floor_mean.csv")),
                      ("No analysis",  Path("results/clean/o3mini_no_analysis_reviewed_v4_floor_mean.csv")),
                      ("Original",     Path("results/clean/o3mini_reviewed_v4_floor_mean.csv"))],
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
        "line_title": "Human vs AI Grades (o3-mini, content) with Error Bars (Floor Mean Review)",
    },
    {
        "exp_name": "cot",
        "model_label": "o3-mini_CoT_experiment",
        "n_runs": 5,
        "summary_csv": Path("results/clean/o3mini_CoT_summary.csv"),
        "out_dir": Path("results/graphs/o3mini_CoT"),
        "name_map": [("o3mini_no_CoT_FM","No CoT"),("o3mini_with_CoT_FM","With CoT")],
        "csv_files": [("No CoT",  Path("results/clean/o3mini_no_CoT_reviewed_v4_floor_mean.csv")),
                      ("With CoT",Path("results/clean/o3mini_reviewed_v4_floor_mean.csv"))],
        "colors": ["#1f77b4", "#ff7f0e"],
        "line_title": "Human vs AI Grades (o3-mini, CoT) with Error Bars (Floor Mean Review)",
    },
    {
        "exp_name": "model",
        "model_label": "model_comparison_experiment",
        "n_runs": 5,
        "summary_csv": Path("results/clean/model_summary.csv"),
        "out_dir": Path("results/graphs/model"),
        "name_map": [("o3mini_FM","o3mini"),("o3_FM","o3"),("gpt-4o_FM","gpt-4o")],
        "csv_files": [("o3mini", Path("results/clean/o3mini_reviewed_v4_floor_mean.csv")),
                      ("o3",     Path("results/clean/o3_reviewed_v4_floor_mean.csv")),
                      ("gpt-4o", Path("results/clean/gpt-4o_reviewed_v4_floor_mean.csv"))],
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
        "line_title": "Human vs AI Grades (model comparison) with Error Bars (Floor Mean Review)",
    },
    {
        "exp_name": "ensemble",
        "model_label": "o3_ensemble_experiment",
        "n_runs": None,
        "summary_csv": Path("results/clean/ensemble_summary.csv"),
        "out_dir": Path("results/graphs/o3_ensemble"),
        "name_map": [("First_attempt","First attempt"), ("Second_attempt","Second attempt"), ("Third_attempt","Third attempt"),
                     ("Fourth_attempt","Fourth attempt"), ("Fifth_attempt","Fifth attempt"),
                     ("o3_mode","Mode, N = 5"), ("o3_floor_mean","Floor mean, N = 5")
                    ],
        "csv_files": [("First attempt",  Path("results/predictions/o3_grader_20250904_114835_fbb8be.csv")),
                      ("Second attempt", Path("results/predictions/o3_grader_20250904_124104_7b7e6d.csv")),
                      ("Third attempt",  Path("results/predictions/o3_grader_20250904_133336_d613f8.csv")),
                      ("Fourth attempt", Path("results/predictions/o3_grader_20250904_142630_896d42.csv")),
                      ("Fifth attempt",  Path("results/predictions/o3_grader_20250904_151418_f0b5eb.csv")),
                      ("Mode, N = 5",   Path("results/clean/o3_reviewed_v4_mode.csv")),
                      ("Floor mean, N = 5", Path("results/clean/o3_reviewed_v4_floor_mean.csv"))
                        ],
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"],
        "line_title": "Human vs AI Grades (o3 ensemble) with Error Bars (various attempts and ensemble methods)",
    },
]

# plotting
REQUIRED_COLS = [
    "source","n","QWK","MAE_points",
    "signed_bias","weighted_bias_pct",
    "mean_abs_gap_pct","WAPE_pct",
    "over_rate","under_rate","avg_std_over_runs",
]

def _title_suffix(exp):
    n = N_RUNS_OVERRIDE if N_RUNS_OVERRIDE is not None else exp["n_runs"]
    return f" — {exp['model_label']}, N={n}"

def plot_summary(exp):
    p = exp["summary_csv"]
    df = pd.read_csv(p)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{exp['exp_name']}] Summary CSV missing columns: {missing}")

    if "equal_rate" not in df.columns:
        df["equal_rate"] = 1 - (df["over_rate"] + df["under_rate"])

    name_map = dict(exp["name_map"])
    labels = [name_map.get(s, s) for s in df["source"]]
    x = np.arange(len(df))
    bw = 0.35

    out_dir = exp["out_dir"]; out_dir.mkdir(parents=True, exist_ok=True)

    # QWK
    fig, ax = plt.subplots(figsize=(8,4), dpi=150, constrained_layout=True)
    ax.bar(x, df["QWK"].to_numpy())
    ax.set_title("Quadratic Kappa (QWK)"+_title_suffix(exp))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(out_dir / "qwk.png", bbox_inches="tight", dpi=300); plt.close(fig)

    # MAE points
    fig, ax = plt.subplots(figsize=(8,4), dpi=150, constrained_layout=True)
    ax.bar(x, df["MAE_points"].to_numpy())
    ax.set_title("MAE (points)"+_title_suffix(exp))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(out_dir / "mae_points.png", bbox_inches="tight", dpi=300); plt.close(fig)

    # MAE %
    if "MAE_pct" in df.columns:
        fig, ax = plt.subplots(figsize=(8,4), dpi=150, constrained_layout=True)
        ax.bar(x, df["MAE_pct"].to_numpy())
        ax.set_title("MAE (percentage)"+_title_suffix(exp))
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.0f}%"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.savefig(out_dir / "mae_pct.png", bbox_inches="tight", dpi=300); plt.close(fig)

    # Error % (mean abs gap vs WAPE)
    fig, ax = plt.subplots(figsize=(8,4), dpi=150, constrained_layout=True)
    ax.bar(x - bw/2, df["mean_abs_gap_pct"].to_numpy(), width=bw, label="Mean Abs Gap %")
    ax.bar(x + bw/2, df["WAPE_pct"].to_numpy(),        width=bw, label="WAPE %")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.0f}%"))
    ax.set_title("Absolute Error (Percent)"+_title_suffix(exp))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False); ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(out_dir / "error_pct.png", bbox_inches="tight", dpi=300); plt.close(fig)

    # Bias %
    fig, ax = plt.subplots(figsize=(8,4), dpi=150, constrained_layout=True)
    ax.bar(x - bw/2, df["signed_bias"].to_numpy(),    width=bw, label="Signed bias %")
    # ax.bar(x + bw/2, df["weighted_bias_pct"].to_numpy(), width=bw, label="Weighted bias %")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.0f}%"))
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Signed Bias (Percent) — +over / −under"+_title_suffix(exp))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False); ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(out_dir / "bias_pct.png", bbox_inches="tight", dpi=300); plt.close(fig)

    # Over/Equal/Under stacked
    fig, ax = plt.subplots(figsize=(8,5), dpi=150, constrained_layout=True)
    over  = (df["over_rate"].to_numpy()  * 100)
    under = (df["under_rate"].to_numpy() * 100)
    equal = (df["equal_rate"].to_numpy() * 100)
    resid = 100 - (under + equal + over)
    equal = np.clip(equal + resid, 0, 100)

    C_UNDER="tab:orange"; C_EQUAL="#BDBDBD"; C_OVER="tab:blue"
    bu = ax.bar(x, under, color=C_UNDER, label="Under-rate", zorder=2)
    be = ax.bar(x, equal, bottom=under, color=C_EQUAL, label="Exact-match", zorder=2)
    bo = ax.bar(x, over,  bottom=under+equal, color=C_OVER,  label="Over-rate", zorder=2)

    def autolabel(bars, vals, bottoms=None):
        if bottoms is None: bottoms = np.zeros_like(vals)
        for rect, v, b in zip(bars, vals, bottoms):
            if v <= 0: continue
            ax.text(rect.get_x()+rect.get_width()/2, b+v/2, f"{v:.2f}%",
                    ha="center", va="center", fontsize=9)

    autolabel(bu, under)
    autolabel(be, equal, bottoms=under)
    autolabel(bo, over,  bottoms=under+equal)

    ax.set_ylim(0,100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.0f}%"))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Over / Exact / Under Rates (stacked)"+_title_suffix(exp))
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    fig.savefig(out_dir / "over_under.png", bbox_inches="tight", dpi=300); plt.close(fig)

    # Stability
    if exp["exp_name"] != "ensemble":
        fig, ax = plt.subplots(figsize=(8,4), dpi=150, constrained_layout=True)
        ax.bar(x, df["avg_std_over_runs"].to_numpy())
        ax.set_title("Stability: Avg Std over Runs (lower = steadier)"+_title_suffix(exp))
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.savefig(out_dir / "stability.png", bbox_inches="tight", dpi=300); plt.close(fig)

    # Bias vs WAPE
    fig, ax = plt.subplots(figsize=(6,5), dpi=150, constrained_layout=True)
    ax.scatter(df["weighted_bias_pct"], df["WAPE_pct"])
    for i, lab in enumerate(labels):
        ax.annotate(lab, (df["weighted_bias_pct"].iat[i], df["WAPE_pct"].iat[i]),
                    textcoords="offset points", xytext=(5,4), fontsize=8)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("Weighted bias %  (+over / −under)")
    ax.set_ylabel("WAPE %  (weighted absolute error)")
    fig.savefig(out_dir / "bias_vs_wape.png", bbox_inches="tight", dpi=300); plt.close(fig)

    print(f"[{exp['exp_name']}] saved summary plots to {out_dir.resolve()}")

def plot_scatter_with_lines(exp):
    pairs = exp["csv_files"]

    fm_map = load_predicted_marks_map(Path("data/SAS-Bench/math_test_student.jsonl"))
    ts_map = load_teacher_score_map(Path("data/SAS-Bench/math_test_teacher.jsonl"))

    rows = []
    for label, path in pairs:
        if not path.exists():
            print(f"[{exp['exp_name']}] missing file {path} (skip {label})")
            continue
        df = _safe_read_csv(path, usecols=["script_id","model_score","std_score"])
        if "script_id" not in df.columns or "model_score" not in df.columns:
            print(f"[{exp['exp_name']}] {path} missing required cols (skip {label})")
            continue
        df["script_id"] = df["script_id"].astype(str)
        df["model_score"] = pd.to_numeric(df["model_score"], errors="coerce")
        df = df.dropna(subset=["model_score"]).drop_duplicates("script_id", keep="last")

        if exp["exp_name"] != "ensemble":
            df["std_score"] = pd.to_numeric(df.get("std_score", 0.0), errors="coerce").fillna(0.0)
        df["method"] = label
        rows.append(df)

    if not rows:
        print(f"[{exp['exp_name']}] nothing to plot for scatter/line")
        return

    long = pd.concat(rows, ignore_index=True)
    long["teacher_score"] = long["script_id"].map(ts_map)
    long["full_marks"]    = long["script_id"].map(fm_map)
    long = long.dropna(subset=["teacher_score","full_marks"]).copy()
    long["teacher_score"] = long["teacher_score"].astype(int)
    long["full_marks"]    = long["full_marks"].astype(int)
    long = long[long["full_marks"] > 0]
    long["teacher_pct"] = (long["teacher_score"] / long["full_marks"]) * 100
    long["model_pct"]   = (long["model_score"]   / long["full_marks"]) * 100

    plt.figure(figsize=(10,7))
    
    color_list = exp.get("colors")
    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    labels_order = [lbl for (lbl, _) in pairs]
    colors = {}
    for i, lbl in enumerate(labels_order):
        if color_list and i < len(color_list):
            colors[lbl] = color_list[i]
        else:
            colors[lbl] = palette[i % max(1, len(palette))]

    jitter = 0.6
    rng = np.random.default_rng(7)

    for method in labels_order:
        d = long[long["method"] == method].copy()
        if d.empty: continue
        x = d["teacher_pct"].to_numpy()
        y = d["model_pct"].to_numpy()
        yerr = d["std_score"].to_numpy()
        xj = np.clip(x + rng.uniform(-jitter, jitter, len(d)), 0, 100)
        yj = np.clip(y + rng.uniform(-jitter, jitter, len(d)), 0, 100)

        plt.scatter(xj, yj, s=22, color=colors[method], alpha=0.35, edgecolors="none")
        plt.errorbar(xj, yj, yerr=yerr, fmt="none", ecolor=colors[method],
                     elinewidth=1, capsize=2, alpha=0.25)

        if len(x) >= 2:
            a, b = np.polyfit(x, y, 1)
            xr = np.linspace(max(0, x.min()), min(100, x.max()), 100)
            yr = a * xr + b
            plt.plot(xr, yr, color=colors[method], linewidth=2, label=f"{method}  (a={a:.2f}, b={b:.2f})")
        else:
            plt.plot([], [], color=colors[method], linewidth=2, label=f"{method}")

    plt.plot([0,100],[0,100], '--', color="grey", linewidth=1.2, label="Ideal line")
    n = N_RUNS_OVERRIDE if N_RUNS_OVERRIDE is not None else exp["n_runs"]
    plt.title(exp["line_title"] + f" — {exp['model_label']}, N={n}")
    plt.xlabel("Human Grade (%)"); plt.ylabel("AI Grade (%)")
    plt.xlim(0,100); plt.ylim(0,100)
    plt.legend(frameon=False, ncols=2)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out = exp["out_dir"]; out.mkdir(parents=True, exist_ok=True)
    save_path = out / "human_vs_ai_scatter.png"
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"[{exp['exp_name']}] saved scatter/line → {save_path.resolve()}")

# runner
def main():
    active_set = set(ACTIVE)
    for exp in EXPERIMENTS:
        if exp["exp_name"] not in active_set:
            continue
        if DO_SUMMARY:
            plot_summary(exp)
        if DO_LINE:
            if exp["exp_name"] != "ensemble":
                plot_scatter_with_lines(exp)

if __name__ == "__main__":
    main()

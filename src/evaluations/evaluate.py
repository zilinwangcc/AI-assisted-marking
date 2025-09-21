from pathlib import Path
import json, ast, re, glob
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
import argparse


# to parse the 'scores' column
def _safe_json(cell):
    if isinstance(cell, dict):
        return cell
    if not isinstance(cell, str):
        return {}
    txt = cell.strip().lstrip("`").rstrip("`")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub("'", '"', txt))
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(txt)
            except Exception:
                return {}


# load predictions CSV
def load_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # check the columns exist
    if "model_score" in df.columns:
        return df

    # compute from scores JSON
    df["scores_dict"] = df["scores"].apply(_safe_json)
    bad = ((df["scores_dict"].apply(len) == 0).sum())
    if bad:
        print(f"skipped {bad} rows with unparsable `scores`")
    df = df[df["scores_dict"].apply(bool)].copy()
    df["model_score"] = df["scores_dict"].apply(lambda d: sum(d.values()))
    return df


# load teacher subset JSONL
def load_teacher_truth(jsonl_path: str | Path) -> pd.DataFrame:
    records = []
    with open(jsonl_path, 'r', encoding='utf8') as f:
        for line in f:
            obj = json.loads(line)
            records.append({
                'script_id':   obj['id'],
                'teacher_score': obj['awarded_marks']
            })
    return pd.DataFrame(records)


# metrics: compare raw model_score vs raw teacher_score, NOT at percent-of-maximum
def compute_metrics(df: pd.DataFrame):
    y_true = df['teacher_score'].astype(int)
    y_pred = df['model_score'].astype(int)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    mae = mean_absolute_error(y_true, y_pred)
    return qwk, mae

# find newest file matching pattern
def newest(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    return files[-1]

# load and merge predictions with teacher truth
def load_and_merge(pred_csv_path=None):
    if pred_csv_path is None:
        pred_csv = newest('results/predictions/*o3_*.csv')
    else:
        pred_csv = pred_csv_path

    print(f"Loading predictions from: {pred_csv}")
    teacher_jsonl = 'data/SAS-Bench/math_test_teacher.jsonl'

    preds = load_predictions(pred_csv)
    truth = load_teacher_truth(teacher_jsonl)

    df = preds.merge(truth, on='script_id', how='left', validate='1:1')
    print(f"Predictions: {Path(pred_csv).name}")
    print(f"Teacher truth: {Path(teacher_jsonl).name}")
    return df


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", nargs="?", default=None, help="Path to predictions CSV")
    args = parser.parse_args()

    df = load_and_merge(args.csv_file)
    qwk, mae = compute_metrics(df)
    print(f"Loaded {len(df)} examples")
    print(f"Quadratic κ  : {qwk:.3f}")
    print(f"Mean Abs Err.: {mae:.3f}")

if __name__ == '__main__':
    main()
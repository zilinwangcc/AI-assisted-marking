from pathlib import Path
import json, re
import pandas as pd


# Paths
simplified_path = "rubrics/math_simplified_rubrics_multi.jsonl"
full_path       = "rubrics/math_initial_rubrics_multi.jsonl"
refined_s_path  = "rubrics/math_refined_simplified_rubrics_multi.jsonl"
refined_f_path  = "rubrics/math_refined_rubrics_multi.jsonl"


# Robust loader
def load_multiblock_json(path: str):
    """
    Loads files containing multiple JSON objects separated by blank lines (jsonl)
    """
    p = Path(path)
    raw = p.read_text(encoding="utf8").strip()

    # Normalise '}\n{' then split on blank lines.
    raw = re.sub(r'\}\s*\n\s*\{', '}\n\n{', raw)
    blocks = [b for b in re.split(r'\n\s*\n', raw) if b.strip()]

    records = []
    for i, block in enumerate(blocks, 1):
        block = block.strip()
        records.append(json.loads(block))

        # Remove trailing commas before } or ]
        cleaned = re.sub(r',\s*([}\]])', r'\1', block)
        records.append(json.loads(cleaned))

        # Wrap 
        candidate = "[" + block.replace("}\n{", "},\n{") + "]"
        arr = json.loads(candidate)
        if isinstance(arr, list):
            records.extend(arr)


    return records

# Summaries & helpers
def summarize_rubrics(rubrics, variant_label: str):
    rows = []
    for r in rubrics:
        criteria = r.get("criteria", []) or []
        num_criteria = len(criteria)
        # sum integer max scores robustly
        total_max = 0
        for c in criteria:
            try:
                total_max += int(c.get("max_score", 0) or 0)
            except Exception:
                pass
        # words in descriptions
        desc_words = sum(len((c.get("description") or "").split()) for c in criteria)
        avg_words = (desc_words / num_criteria) if num_criteria else 0.0
        # verbosity proxy
        rubric_text = (
            (r.get("reference") or "") + " " + (r.get("granularity") or "") + " " +
            " ".join((c.get("description") or "") + " " + str(c.get("partial_credit") or "")
                     for c in criteria)
        ).strip()
        words_total = len(rubric_text.split()) if rubric_text else 0
        chars_total = len(rubric_text)

        rows.append({
            "question_id": r.get("question_id"),
            "variant": variant_label,
            "num_criteria": num_criteria,
            "total_max_score": total_max,
            "avg_words_per_criterion": round(avg_words, 2),
            "rubric_words_total": words_total,
            "rubric_chars_total": chars_total,
        })
    return pd.DataFrame(rows)

def aggregate_table(df, label):
    if df.empty:
        return pd.DataFrame([{"variant": label}])
    return pd.DataFrame([{
        "variant": label,
        "avg_num_criteria": round(float(df["num_criteria"].mean()), 2),
        "median_num_criteria": int(df["num_criteria"].median()),
        "avg_words_per_criterion_mean": round(float(df["avg_words_per_criterion"].mean()), 2),
        "avg_words_per_criterion_std": round(float(df["avg_words_per_criterion"].std(ddof=0)), 2),
        "rubric_words_total_mean": round(float(df["rubric_words_total"].mean()), 1),
    }])


# Load all four variants
rubrics_s  = load_multiblock_json(simplified_path)
rubrics_f  = load_multiblock_json(full_path)
rubrics_sr = load_multiblock_json(refined_s_path)
rubrics_fr = load_multiblock_json(refined_f_path)

df_s  = summarize_rubrics(rubrics_s,  "simplified")
df_f  = summarize_rubrics(rubrics_f,  "full")
df_sr = summarize_rubrics(rubrics_sr, "simplified_refined")
df_fr = summarize_rubrics(rubrics_fr, "full_refined")


# Aggregate overview
agg = pd.concat([
    aggregate_table(df_s,  "simplified"),
    aggregate_table(df_sr, "simplified_refined"),
    aggregate_table(df_f,  "full"),
    aggregate_table(df_fr, "full_refined"),
], ignore_index=True)

print(agg)


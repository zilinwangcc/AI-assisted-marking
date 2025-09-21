import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import csv

# Setup
load_dotenv()
client = OpenAI()

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "SAS-Bench" / "math_test_student.jsonl"
RUBRIC_PATH = ROOT / "rubrics" / "math_initial_rubrics_multi.jsonl"
OUT_PATH = ROOT / "results" / "rubric_quality_review_full.csv"


# Load Dataset and Rubrics
def load_multiblock_json(path):
    """
    Read a file that contains multiple JSON objects (JSONL)
    Returns a list[dict].
    """
    raw = Path(path).read_text(encoding="utf8").strip()
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    records = []
    for block in blocks:
        records.append(json.loads(block))
    return records

def load_jsonl(path):
    """
    Read a JSON Lines file into a list[dict], one object per line.
    """
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f]

data = load_jsonl(DATASET_PATH)
rubrics = load_multiblock_json(RUBRIC_PATH)


# Group by question ID
id_to_question = {d["id"]: d["question"] for d in data}
id_to_analysis = {d["id"]: d.get("analysis", "") for d in data}

question_to_rubrics = {}
for r in rubrics:
    # search the question text by rubric's question_id
    q = id_to_question.get(r["question_id"])
    if not q:
        continue
    if q not in question_to_rubrics:
        question_to_rubrics[q] = {
            "rubrics": [],
            "analysis": id_to_analysis.get(r["question_id"], "")
        }
    question_to_rubrics[q]["rubrics"].append(r)


def build_eval_prompt(question, rubric_obj, analysis):
    """
    Create the prompt for the LLM to evaluate a single rubric.
    - Formats rubric criteria with max_score and partial_credit.
    - Provides the original question and the worked-solution ANALYSIS for context.
    """
    rubric_text = "\n".join(
        f"- [{c['id']}] {c['description']} (Max Score: {c['max_score']})"
        f"\n    Partial Credit: {c.get('partial_credit', 'None')}"
        for c in rubric_obj["criteria"]
    )
    return [
        {"role": "system", "content": "You are a senior examiner and exam designer."},
        {"role": "user", "content": (
            "Here is a math exam question, an official teacher analysis of the solution, "
            "also think about how the rubric will be used, "
            "and what possible student answers and mistakes might be.\n\n"
            "Rate EACH dimension below on a 0-20 scale (integers only), "
            "where 0=very poor, 20=excellent.\n"
            "Dimensions: clarity, coverage, consistency, completeness, fairness.\n"
            "Also return an overall_score (0-100). "
            "You may compute it as the sum of the five 0-20 scores.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"TEACHER ANALYSIS:\n{analysis}\n\n"
            f"RUBRIC:\n{rubric_text}"

            f"Respond in JSON with:\n"
            f"The question, rubric_id\n"
            f"- overall_score (number, 0-100)\n"
            f"- 'scores': a dictionary with numeric ratings from 1 to 5 for each dimension: "
            f"clarity, coverage, consistency, completeness, fairness\n"
            f"- 'comments': brief explanation for each score\n"
            f"- 'recommendations': any suggested improvements\n"
            f"- 'confidence': how confident you are in this evaluation (0.0 to 1.0)\n"
        )}
    ]

# Run LLM
results = []

"""
Loop over each question, and to evaluate the first rubric for that question only.
As the scripts in the same item share the same rubric.
"""
for q, group in tqdm(question_to_rubrics.items()):
    first_rubric = group["rubrics"][0]
    analysis = group["analysis"]
    messages = build_eval_prompt(q, first_rubric, analysis)
    # print(messages)

    try: #call LLM
        resp = client.chat.completions.create(
            model="o3-mini",
            messages=messages,
            temperature=1,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        score = float(data.get("overall_score"))
    except Exception as e:
        content = f"ERROR: {e}"
        score = None

    # Append one row per item
    results.append({
        "question": q,
        "rubric_id": first_rubric["question_id"],
        "score": score,
        "review": content
    })

# Save Results
pd.DataFrame(results).to_csv(OUT_PATH, index=False, quoting=csv.QUOTE_ALL)
print(f"Saved rubric quality review to {OUT_PATH}")

import json
import pandas as pd
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Setup
load_dotenv()
client = OpenAI()

# change the RUBRIC_PATH, REVIEW_PATH, OUT_PATH for other variants
ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "SAS-Bench" / "math_test_student.jsonl"
RUBRIC_PATH = ROOT / "rubrics" / "math_initial_rubrics_multi.jsonl"
REVIEW_PATH = ROOT / "results" / "rubric_quality_review_full.csv"
OUT_PATH = ROOT / "rubrics" / "math_refined_rubrics_multi.jsonl"


# Load Dataset and Rubrics
def load_jsonl(path):
    """
    Read a JSON Lines file into a list[dict], one object per line.
    """
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f]

def load_multiblock_json(path):
    """
    Read a file that contains multiple JSON objects (JSONL)
    Returns a list[dict].
    """
    raw = Path(path).read_text(encoding="utf8").strip()
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    records = []
    for block in blocks:
        try:
            records.append(json.loads(block))
        except Exception as e:
            print(f"Skipping block due to error: {e}")
    return records

# JSON Recovery 
def try_fix_json(raw):
    """
    Attempt to parse 'raw' as JSON.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            fixed = re.search(r"\{.*\}", raw, re.DOTALL).group(0)
            return json.loads(fixed)
        except Exception:
            return None

# Load Data
data = load_jsonl(DATASET_PATH)
rubrics = load_multiblock_json(RUBRIC_PATH)
review_df = pd.read_csv(REVIEW_PATH)

id_to_question = {d["id"]: d["question"] for d in data}
id_to_analysis = {d["id"]: d.get("analysis", "") for d in data}

# search rubric by question_id
question_to_ids = {}
for d in data:
    question = d["question"]
    question_to_ids.setdefault(question, []).append(d["id"])

question_to_review = {
    row["question"]: row["review"]
    for _, row in review_df.iterrows()
    if isinstance(row["review"], str) and not row["review"].startswith("ERROR")
}

# find the full marks for each question
question_to_fullmarks = {}
for d in data:
    fm = d.get("full_marks")
    if fm is not None and d["question"] not in question_to_fullmarks:
        question_to_fullmarks[d["question"]] = int(fm)

# Create prompt
def build_refine_prompt(question, rubric_obj, analysis, review, full_marks):
    """
    Refinement prompt for the LLM.
    - Includes: original question, worked-solusoin analysis, CURRENT rubric, review feedback,
      and total marks (LLM should keep criteria consistent with full marks).
    - keep structure/fields as-is; only update 'criteria'.
    """
    return [
        {"role": "system", "content": "You are a senior exam designer and grader."},
        {"role": "user", "content": (
            "Please revise the rubric JSON below based on the review feedback and the teacher analysis. "
            "Keep the format exactly the same, including fields like 'question_id', "
            "'reference', 'granularity', and 'criteria'.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"WORKED-SOLUTION ANALYSIS:\n{analysis}\n\n"
            f"CURRENT RUBRIC JSON:\n{json.dumps(rubric_obj, ensure_ascii=False, indent=2)}\n\n"
            f"REVIEW FEEDBACK:\n{review}\n\n"
            f"TOTAL MARKS AVAILABLE FOR THIS QUESTION: {full_marks}.\n"
            "Return ONLY the improved JSON object, keeping the same structure and fields as the original rubric.\n"
            "You can update the criteria section based on the feedback and your judgement directly. \n"
            "Revise each criterion in place — do not merge unrelated issues or add new fields.\n"
            "DO NOT add fields like 'preamble', 'legend', or any metadata not originally present.\n\n"
            "Only update the content of 'criteria' based on the review feedback and teacher analysis."

        )}
    ]

# LLM Refinement
refined_rubrics = []
with open(OUT_PATH, "w", encoding="utf8") as f:
    # Iterate per question; create ONE improved rubric, then replicate for all ids.
    for question, ids in tqdm(question_to_ids.items()):
        if question not in question_to_review:
            print(1)
            continue

        rubric_obj = next((r for r in rubrics if r.get("question_id") == ids[0]), None)
        if not rubric_obj:
            continue

        analysis = id_to_analysis.get(ids[0], "")
        review = question_to_review[question]
        full_marks  = question_to_fullmarks.get(question)

        messages = build_refine_prompt(question, rubric_obj, analysis, review, full_marks)
        # print(messages)

        try: # call LLM
            response = client.chat.completions.create(
                model="o3-mini",
                messages=messages,
                temperature=1,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            new_rubric = try_fix_json(content)
            if not new_rubric:
                raise ValueError("Unable to parse JSON response.")
        except Exception as e:
            print(f"Error refining rubric for question: {question[:60]}...\n{e}")
            continue

        # Replicate the refined rubric across ALL script ids for this question.
        for id_ in ids:
            new_rubric["question_id"] = id_
            f.write(json.dumps(new_rubric, ensure_ascii=False, indent=2))
            f.write("\n\n")

print(f"Saved refined rubrics to {OUT_PATH}")

import json
import csv
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError
from tqdm import tqdm

# setup 
ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "results" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
client = OpenAI()

# Dataset loader 
def load_dataset(jsonl_path: str, rubric_path: str) -> pd.DataFrame:

    #load rubrics
    rubric = {}
    raw = Path(rubric_path).read_text(encoding="utf8").strip()
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    for block in blocks:
        rec = json.loads(block)
        qid = rec.get("id") or rec.get("question_id")
        rubric[qid] = rec

    # load the student subset 
    rows = []
    for line in open(jsonl_path, encoding="utf8"):
        obj      = json.loads(line)
        meta     = rubric[obj["id"]]
        criteria = meta["criteria"]
        rows.append({
            "script_id":   obj["id"],
            "question":    obj.get("question"),
            "criteria":    criteria,
            "responses":   [s["response"] for s in obj.get("steps", [])],
            "manual_label": obj.get("manual_label"),
            "steps":       obj.get("steps", []),
            "analysis":  obj["analysis"],
            "reference": obj.get("reference", ""),
        })
    return pd.DataFrame(rows)


# Prompt
def build_messages(row, persona, style, examples_by_question):
    """
    Build the prompt list for one grading request.
    Includes:
      - system persona & instructions (CoT or no-CoT version — CoT currently enabled),
      - intro with question/reference/rubric,
      - the target student's answer,
      - (k-shot)optional assistant message with teacher-marked exemplars,
      - to test with the effect of the in-text info, 
        comment / uncomment out the corresponding lines (for analysis / reference). 
      - specifies JSON-only output format.
    """

    #Rubric criteria
    crit_text = "\n".join(
        f"- “{c['id']}” (max {c['max_score']} pts): {c['description']}; partial credit: {c['partial_credit']}"
        for c in row["criteria"]
    )
    total_max = sum(c["max_score"] for c in row["criteria"])

    #System message: persona + instruction
    system_msg = {
        "role": "system",

        # CoT
        "content": (
            f"You are a {persona} exam grader.  "
            "You will be shown a question, a detailed rubric, the reference answer, "
            "the work-solution analysis, and the student's answer.  "
            "First, think through each rubric criterion one by one, deciding how many points "
            "the student deserves for *that* criterion (no greater than the max paritial scores).  "
            "Then output exactly one JSON object "
            "with those scores.  Do not output any other text.\n"
            "If you are in one-shot mode, you will also be shown an example of how"
            " the teacher graded a worked solution for the same question but different student answer. "
            "If you are in few-shot mode, you will be shown two such examples.  "
            "Use that as a guide for your grading.\n"
            "Use your chain of thought to reflect on the rubric and the student's answer, "
            "and adjust your scores accordingly.\n\n"
            "Output only one JSON object with keys: 'scores', 'comments', and 'confidence'"
            " (float from 0 to 1, your self-confidence on your decision).\n"
        )

        # no CoT
        # "content": (
        #     f"You are a {persona} exam grader. "
        #     "You will be shown a question, a detailed rubric, the reference answer, "
        #     "the work-solution analysis, and the student's answer. \n\n"
        #     "Decide the points for each rubric criterion and 
        #     "then output exactly one JSON object.\n "
        #     "If you are in one-shot mode, you will also be shown an example of how "
        #     "the teacher graded a worked solution for the same question but different student answer. "
        #     "If you are in few-shot mode, you will be shown two such examples.  "
        #     "Use that as a guide for your grading.\n"
        #     "Do not write your chain-of-thought, analysis, or step-by-step reasoning. \n"
        #     "Provide only the required JSON with brief, surface-level comments.\n\n"
        #     "Output only one JSON object with keys: 'scores', 'comments', and 'confidence' "
        #     "(float from 0 to 1, your self-confidence on your decision)\n."
        # )
    }

    # User message: question + rubric
    user_intro = {
        "role": "user",
        "content": (
            f"QUESTION:\n{row['question']}\n\n"
            f"The reference solution:\n{row['reference']}\n\n"
            f'The work-solution analysis is:\n{row["analysis"]}\n\n'
            f"(Total points available: {total_max})\n\n"
            "RUBRIC (follow exactly):\n"
            f"{crit_text}\n\n"

            "STUDENT ANSWER:"
        )
    }

    # User message: the student’s answer
    student_ans = {
        "role": "user",
        "content": "\n".join(row["responses"])
    }

    messages = [system_msg, user_intro, student_ans]

    # Optionally include exemplars (one-shot / few-shot)
    if style in ("one_shot", "few_shot"):
        exemplars = examples_by_question.get(row["question"], [])
        exemplars = exemplars[:1] if style == "one_shot" else exemplars[:2]

        if exemplars:
            example_student_ans = []
            for ex in exemplars:
                ex_id = ex.get("id", "UNKNOWN_ID")
                steps = ex.get("steps", [])
                if not isinstance(steps, list):
                    steps = []
                step_lines = []
                for i, s in enumerate(steps, start=1):
                    resp = s.get("response", "")
                    label = s.get("label", "")
                    errors = s.get("errors", "")
                    step_lines.append(f'Step {i}: "{resp}" // label: {label}, errors: {errors}')
                example_student_ans.append(f"Example ID {ex_id} (same question, different student):\n" + "\n".join(step_lines))

            # place exemplars as an assistant message
            messages.append({
                "role": "assistant",
                "content": (
                    "Here are teacher-marked exemplars to guide partial-credit decisions:\n\n"
                    "The label is the partial-credit score assigned of this step by the teacher, "
                    "and 'errors' shows the explanation of why the teacher made their decision.\n\n"
                    + "\n\n".join(example_student_ans)
                )
            })


    # grading request & output schema
    messages.append({
        "role": "user",
        "content": (
            "Now grade the *student's* answer.  "
            "For each rubric criterion, assign an integer score ≤ its max.  "
            f"The sum must not exceed {total_max}.\n\n"
            "Output exactly one JSON object and nothing else, in this shape:\n"
            "{\n"
            '  "scores": { "<criterion_id>": <points>, … }\n'
            '  "comments": { "<criterion_id>": "<brief explanation>", … }\n'
            "}\n"
            "Use the 'comments' field to explain your decision for each criterion. "
        )
    })

    return messages



# Main grading loop
def grade_run(
    dataset_path: str,
    rubric_path: str,
    persona: str = "strict",
    model: str   = "gpt-4o",
    prompt_style: str = "one_shot",
    temperature: float = 0.0
):
    """
    Run a single-pass grading job over all scripts.
    Writes a CSV in results/predictions with per-script scores/comments.
    """
    # Load dataset
    df = load_dataset(dataset_path, rubric_path)

    EXAMPLE_PATH = Path("data/SAS-Bench/math_rubrics_student.jsonl")
    example_df = pd.read_json(EXAMPLE_PATH, lines=True)

    examples_df = example_df[["id", "question", "analysis", "steps"]].copy()

    examples_df = examples_df.sort_values(["question", "id"])

    # Build {question: [rows]} with 1 exemplar for one-shot, 2 for few-shot
    if prompt_style == "one_shot":
        limited = examples_df.groupby("question", as_index=False).head(1)
    elif prompt_style == "few_shot":
        limited = examples_df.groupby("question", as_index=False).head(2)
    else:  # zero_shot
        limited = examples_df.iloc[0:0]  # empty

    examples_by_question = (
        limited.groupby("question")
        .apply(lambda g: g.to_dict("records"))
        .to_dict()
    )


    # Prepare output CSV
    run_id   = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    csv_name = f"o3_grader_{run_id}.csv"
    out_path = PRED_DIR / csv_name

    rows_out = []
    # Iterate over each student response
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grading"):
        messages = build_messages(row, persona, prompt_style, examples_by_question)
        # print(messages)

        try:
            # call llm
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type":"json_object"}
            )
            content = resp.choices[0].message.content
            usage = resp.usage.total_tokens
        except (RateLimitError, APIConnectionError) as e:
            print("Skipping:", e)
            continue

        # Parse & record
        try:
            result = content if isinstance(content, dict) else json.loads(content)
            # Compute a total score from per-criterion scores
            result["model_score"] = sum(result.get("scores", {}).values())
            result["script_id"]   = row["script_id"]
            result["tokens_used"] = usage
            rows_out.append(result)
        except Exception as e:
            print("Skipping:", e)
            continue

    # Write output CSV
    if rows_out:
        all_keys = set().union(*(r.keys() for r in rows_out))
        fieldnames = ["script_id", "scores", "model_score", "comments"] \
                   + sorted(all_keys - {"script_id","scores","model_score","comments"})
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"Wrote {len(rows_out)} rows → {out_path}")
    else:
        print("Nothing written.")


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rubric",  required=True)
    ap.add_argument("--persona", default="strict")
    ap.add_argument("--model",   default="o3")
    ap.add_argument("--prompt_style",choices=["zero_shot","one_shot","few_shot"],default="one_shot")
    ap.add_argument("--temperature",type=float,default=1)
    args=ap.parse_args()
    grade_run(
        args.dataset,args.rubric,
        args.persona,args.model,
        args.prompt_style,args.temperature
    )
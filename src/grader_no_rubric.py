import json, re, uuid
from pathlib import Path
from datetime import datetime
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import ast


# Setup
ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "results" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
client = OpenAI()

# Dataset Loading
def load_dataset(jsonl_path: str) -> pd.DataFrame:
    # load the student subset
    rows = []
    for line in open(jsonl_path, encoding="utf8"):
        obj = json.loads(line)
        rows.append({
            "script_id": obj["id"],
            "question": obj.get("question"),
            "responses": [s["response"] for s in obj.get("steps", [])],
            "analysis": obj.get("analysis"),
            "steps": obj.get("steps", []),
            "full_marks": obj.get("full_marks"),
            "reference": obj.get("reference", ""),
        })
    return pd.DataFrame(rows)


def normalise_result_shape(result, crit_ids):
    """
    Force model output into: {'scores': {cid:int,...}, 
    'comments': {cid:str,...}, 'confidence': float|None}.
    """
    if not isinstance(result, dict):
        return None

    # ---- scores ----
    scores = result.get("scores", {})

    if isinstance(scores, (int, float)): # single number
        scores = {cid: 0 for cid in crit_ids}
        scores[crit_ids[-1]] = int(scores.pop(crit_ids[-1], 0)) + int(result["scores"])

    elif isinstance(scores, list): # list       
        mapped = {}
        for cid, v in zip(crit_ids, scores):
            mapped[cid] = int(v) if isinstance(v, (int, float)) else 0
        for cid in crit_ids[len(mapped):]:
            mapped[cid] = 0
        scores = mapped

    elif isinstance(scores, dict): # dict
        scores = {cid: int(scores.get(cid, 0) or 0) for cid in crit_ids}

    else:
        scores = {cid: 0 for cid in crit_ids}

    # comments
    comments = result.get("comments", {})
    if not isinstance(comments, dict):
        comments = {}
    comments = {cid: str(comments.get(cid, "")) for cid in crit_ids}

    # confidence
    conf = result.get("confidence", None)
    try:
        conf = float(conf) if conf is not None else None
    except Exception:
        conf = None

    return {"scores": scores, "comments": comments, "confidence": conf}


# Prompt Builder
def parse_model_json(raw: str) -> dict:
    """
    Parse model output into JSON robustly
      (no code fences, no trailing commas, normalises quotes).
    """
    if raw is None:
        raise ValueError("Empty model output")

    txt = raw.strip()

    # Remove code fences if present
    txt = re.sub(r"^\s*```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```\s*$", "", txt)

    # Normalize quotes in keys
    txt = txt.replace("“", '"').replace("”", '"')

    # Remove trailing commas before } or ]
    txt = re.sub(r",(\s*[}\]])", r"\1", txt)

    # Try strict JSON first
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(txt)
            json.dumps(obj)
            return obj
        except Exception as e:
            raise ValueError(f"JSON decoding failed after cleanup: {e}\nRaw content: {raw[:2000]}")


def build_messages(row, persona, style, examples_by_question):
    """
    Build a no-rubric grading prompt for a single script.
    """
    total_max = row['full_marks']

    messages = [
        {
        "role": "system",
        "content": (
            f"You are a very {persona} and harsh exam grader.\n"
            "You will be shown a question, its reference answer, "
            "work-solution analysis, and the student's answer.\n"
            "No rubric is available. You must assign marks based on correctness, "
            "logical reasoning, and completeness.\n\n"
            f"The total marks available is {total_max}.\n\n"
            "IMPORTANT:\n"
            "- Do NOT use LaTeX, markdown, or symbols like `\\( ... \\)`, `{`, `\\lambda`, etc.\n"
            "- Write plain-text feedback for each criterion.\n"
            "- Output one JSON object with exactly these keys: `scores`, `comments`, and `confidence`."
            )
        },
    ]

    # print("\n\n\n\n\n\n\n\n\n\n", "--------------------\n\n\n\n\n\n\n\n\n\n")
    # Final grading instruction
    messages.append({
        "role": "user",
        "content": (
            f"QUESTION:\n{row['question']}\n\n"
            f"(Total points available: {total_max})\n\n"
            f"The reference solution:\n{row['reference']}\n\n"
            f'The work-solution analysis is:\n{row["analysis"]}\n\n'
            "STUDENT ANSWER:" + "\n".join(row["responses"])
            + "\n\n" 
        ),
    })

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
                example_student_ans.append(
                    f"Example ID {ex_id} (same question, different student):\n" 
                    + "\n".join(step_lines))

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
            "Now grade the *student’s* answer.\n"
            f"Assign integer scores (≤ total of {total_max}) using the criterion IDs C1, C2...Cn\n"
            "Output JSON only, in this format:\n"
            "{\n"
            '  "scores": { "C1": <int>, "C2": <int>, ... },\n'
            '  "comments": { "C1": "reason", "C2": "reason", ... },\n'
            '  "confidence": <float between 0 and 1>\n'
            "}"
        )
    })

    return messages




# Main Runner
def grade_run(
    dataset_path, 
    persona="neutral", 
    model="o3-mini", 
    temperature=0.0, 
    prompt_style="one_shot"):
    
    df = load_dataset(dataset_path)
    example_df = pd.read_json("data/SAS-Bench/math_rubrics_student.jsonl", lines=True)

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

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    out_path = PRED_DIR / f"o3mini_no_rubrics_grader_{run_id}.csv"

    rows_out = []
    for _, row in df.iterrows():
        messages = build_messages(row, persona, prompt_style, examples_by_question)
        # print(messages)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                # stream=False,
                response_format={"type": "json_object"}, 
            )
            content = resp.choices[0].message.content
            usage = resp.usage.total_tokens
            # print("1,",  usage)
            # logprob_info = json.loads(resp.choices[0].logprobs.model_dump_json())
        except Exception as e:
            print(f"Error grading {row['script_id']}: {e}")
            continue

        try:
            content = resp.choices[0].message.content
            result = parse_model_json(content)

            crit_ids = [c["id"] for c in row.get("criteria", [])] or ["C1","C2","C3"]

            norm = normalise_result_shape(result, crit_ids)
            if norm is None:
                # dump & skip cleanly
                Path("bad_outputs").mkdir(parents=True, exist_ok=True)
                with open(f"bad_outputs/{row['script_id']}.raw.txt", "w", encoding="utf8") as f:
                    f.write(str(content))
                print(f"Parse error for {row['script_id']}: could not coerce model output; dumped raw JSON.")
                continue

            scores    = norm["scores"]
            comments  = norm["comments"]
            confidence = norm["confidence"]

            rows_out.append({
                "script_id": row["script_id"],
                "confidence": confidence,
                "model_score": sum(scores.values()),
                # "score_probs": json.dumps(score_probs),
                "scores": json.dumps(scores),
                "comments": json.dumps(comments),
                "tokens_used": usage,
            })
        except Exception as e:
            print(f"Parse error for {row['script_id']}: {e}")
            continue

    if rows_out:
        pd.DataFrame(rows_out).to_csv(out_path, index=False)
        print(f"Wrote {len(rows_out)} rows to {out_path}")
    else:
        print("No output.")


# ─── CLI Entrypoint ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--persona", default="neutral")
    ap.add_argument("--model", default="o3-mini")
    ap.add_argument("--temperature", type=float, default=1)
    ap.add_argument("--prompt_style", choices=["zero_shot", "one_shot", "few_shot"], default="zero_shot")
    args = ap.parse_args()

    grade_run(
        dataset_path=args.dataset,
        persona=args.persona,
        model=args.model,
        temperature=args.temperature,
        prompt_style=args.prompt_style
    )

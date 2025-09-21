import json
from pathlib import Path
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATASET_PATH = Path("data/SAS-Bench/cleaned_Math_ShortAns.jsonl")
EXEMPLAR_PATH = Path("rubrics/example_rubric.jsonl")

# Change the OUTPUT_PATH as needed
OUTPUT_PATH = Path("rubrics/math_initial_rubrics_multi.jsonl")
MODEL = "o3-mini"
SAMPLE_PER_Q = None

# Initialise OpenAI client
client = OpenAI()

# Load exemplar rubric -> the example rubric from gcse
with open(EXEMPLAR_PATH, encoding='utf8') as f:
    exemplar = json.load(f)

# Load and group all student examples by question text
with open(DATASET_PATH, encoding='utf8') as f:
    entries = [json.loads(line) for line in f if line.strip()]

question_groups = defaultdict(list)
for e in entries:
    question_groups[e['question']].append(e)

rubric_batches = []
rubric_to_ids = {}

for question_text, group in question_groups.items():

    # double check least 4 samples to form a batch
    if len(group) < 4:
        continue
    
    """
    the scripts are already shuffled in the previous step.
    For simplified rubircs: use train_samples = group[:1].
    """
    train_samples = group[:2] # first 2 for the exemplar subset -> full rubric
    test_samples = group[2:] # the rest for test samples

    base_reference = train_samples[0]['reference']
    ideal_solution = train_samples[0]['analysis']

    rubric_batches.append((train_samples, question_text, base_reference, ideal_solution))

    test_ids = [e['id'] for e in test_samples]
    rubric_to_ids[question_text] = test_ids


def trim_steps(steps, k):
    "Return first k steps if k is not None, otherwise return all steps."
    return steps if k is None else steps[:k]


def build_prompt(entries_for_q, question_text, reference, ideal_solution, exemplar):
    """
    Create the rubric generation prompt.
    Uses:
      - exemplar rubric from gcse (format/style)
      - the Chinese question + its final answer 
        + worked-solution anaylysis (ideal solution)
      - one / two example student solutions with step-wise labels/errors
    Returns a single user message string.
    """
    exemplar_block = json.dumps(exemplar, ensure_ascii=False, indent=2)

    student_sections = []
    for entry in entries_for_q:
        header = (
            f"The TOTAL MARKS AVAILABLE for this question is {entry.get('total')}.\n\n"
            f"Here is the '{entry['id']}' example student solution "
            f"which has awarded {entry.get('manual_label')}/{entry.get('total')} by the teacher:"
        )
        
        steps = entry.get('steps', [])
        few = "\n".join(
            f"Step {i+1}:\n  response: {s['response']}\n  label: {s['label']}\n  errors: {s['errors']}"
            for i, s in enumerate(trim_steps(steps, SAMPLE_PER_Q))
        )
        student_sections.append(header + "\n" + few)

    prompt_parts = [
        "Here's an *example rubric* for a question:",
        exemplar_block,
        "Use the provided example rubrics from other questions as a guide"
        " for the desired format and level of detail.",
        
        "\nYou will be given a question in Chinese, its final answer, "
        "an ideal solution (worked-solution analysis; scoring guideline)"

        # change the 'two' to 'one' for the simplified rubric generation
        " with full marks, and two corresponding student answers with step-by-step ground truth teacher grading.",
        "All the partial marks need to be an integer, and the sum of all partial marks"
        " must equal to the full marks available.",
        "Now create a *new rubric* for the following question. "
        "For each step, describe what constitutes a 'Correct', 'Partially Correct', and 'Incorrect' attempt:",
        "\nQUESTION:",
        question_text,
        "\nHere is the FINAL ANSWER of the question:",
        reference,
        "\nHere is an IDEAL SOLUTION (the worked-solution analysis):",
        ideal_solution,
    ]
    prompt_parts.extend(student_sections) # the in-text examples
    prompt_parts.append(
        f"\nReply *only* with JSON and English: an object with keys 'question_id', 'reference',"
          f"'granularity', and 'criteria' — where 'criteria' is an array of 3-5 items"
          f" (or what you think is suitable) with keys 'id', 'description', 'max_score', and 'partial_credit'...\n\n"
    )
    return "\n\n".join(prompt_parts)


with open(OUTPUT_PATH, 'w', encoding='utf8') as out_f:

    # Iterate over per-question batches prepared upstream
    for train_samples, question_text, reference, ideal_solution in rubric_batches:
        
        # Build user prompt
        prompt = build_prompt(train_samples, question_text, reference, ideal_solution, exemplar)
        # print(prompt)

        response = client.chat.completions.create(
            model=MODEL,
            temperature=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a senior examiner and exam designer."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content

        # Parse the JSON string from the model into a Python dict
        rubric_obj = json.loads(content)

        # All remaining scripts (same question) receive the same generated rubric
        test_ids = rubric_to_ids.get(question_text, [])

        for test_id in test_ids:
            # Construct rubric JSON record for this specific test script id
            rubric = {
                "question_id": test_id,
                "reference": reference,
                "granularity": "fine",
                "criteria": rubric_obj["criteria"]
            }
            out_f.write(json.dumps(rubric, ensure_ascii=False, indent=2))
            out_f.write("\n\n")

print(f"Rubrics written to {OUTPUT_PATH}")

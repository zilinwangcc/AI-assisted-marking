import json
from pathlib import Path
import random

# configuration
DATASET_PATH   = Path("data/SAS-Bench/cleaned_Math_ShortAns.jsonl")
SUBSET_COUNT   = None  # global cap on lines

# output paths for student and teacher records
STUDENT_OUT_PATH = Path("data/SAS-Bench/math_rubrics_student.jsonl")
TEACHER_OUT_PATH = Path("data/SAS-Bench/math_rubrics_teacher.jsonl")

# output rubric paths
RUBRICS_DIR = Path("rubrics")

# record holders
rubric_records   = []
student_records  = []
teacher_records  = []


def main():
    student_records = []
    teacher_records = []
    rubric_records  = []

    student_test_records = []
    teacher_test_records = []

    with open(DATASET_PATH, 'r', encoding='utf8') as infile:
        lines = [line for line in infile if line.strip()]
    grouped = {}

    selected_lines = lines[:SUBSET_COUNT] if SUBSET_COUNT else lines

    for line in selected_lines:
        entry = json.loads(line)
        grouped.setdefault(entry['question'], []).append(entry)

    new_id = 0
    for question, entries in grouped.items():
        if len(entries) < 4:
            continue  # to double check, still skip questions with fewer than 4 samples
        
        random.shuffle(entries)
        block = entries[:2]         # first 2 examples go to training
        skipped = entries[2:]       # rest go to test set

        # Process training (block) entries (the exemplar subset)
        for e in block:
            new_entry = e.copy()
            new_entry['id'] = f"Math_ShortAns_{new_id}"
            total = new_entry.get('total', new_entry.get('full_marks'))

            student_records.append({
                'id':          new_entry['id'],
                'question':    new_entry['question'],
                'reference':   new_entry.get('reference'),
                'full_marks':  total,
                'responses':   new_entry.get('steps', []),
                'steps':       new_entry.get('steps', []),
                'analysis':    new_entry.get('analysis', ''),
                'manual_label': new_entry.get('manual_label', 0)
            })
            teacher_records.append({
                'id':            new_entry['id'],
                'analysis':      new_entry.get('analysis', ''),
                'awarded_marks': new_entry.get('manual_label', 0),
                'steps':         new_entry.get('steps', [])
            })
            rubric_records.append({
                'question_id': new_entry['id'],
                'reference':   new_entry.get('reference'),
                'full_marks':  total,
                'criteria':   {new_entry.get('reference'): total}
            })
            new_id += 1

        # Process testing (skipped) entries
        for e in skipped:
            new_entry = e.copy()
            new_entry['id'] = f"Math_ShortAns_{new_id}"
            total = new_entry.get('total', new_entry.get('full_marks'))

            student_test_records.append({
                'id':          new_entry['id'],
                'question':    new_entry['question'],
                'reference':   new_entry.get('reference'),
                'full_marks':  total,
                'responses':   [s['response'] for s in new_entry.get('steps', [])],
                'steps':       new_entry.get('steps', []),
                'analysis':    new_entry.get('analysis', ''),
                'manual_label': new_entry.get('manual_label', 0)
            })
            teacher_test_records.append({
                'id':            new_entry['id'],
                'analysis':      new_entry.get('analysis', ''),
                'awarded_marks': new_entry.get('manual_label', 0),
                'steps':         new_entry.get('steps', [])
            })
            new_id += 1

    # Save outputs
    with open(STUDENT_OUT_PATH, 'w') as f:
        for r in student_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(TEACHER_OUT_PATH, 'w') as f:
        for r in teacher_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open('data/SAS-Bench/math_test_student.jsonl', 'w') as f:
        for r in student_test_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open('data/SAS-Bench/math_test_teacher.jsonl', 'w') as f:
        for r in teacher_test_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'Training: {len(student_records)} students, {len(teacher_records)} teachers')
    print(f'Testing: {len(student_test_records)} students, {len(teacher_test_records)} teachers')


if __name__ == '__main__':
    main()
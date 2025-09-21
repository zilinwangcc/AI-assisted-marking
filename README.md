## Environment

* **OS**: macOS 14.x
* **Arch**: `arm64`
* **Python**: `3.11.5` (Python 3.11.5 (main, Sep 11 2023) [Clang 14.0.6])
* **Virtual env**: `.venv/` (recommended)

---

## API key

Create a `.env` in the repo root:
OPENAI_API_KEY=YOUR_OWN_KEY_HERE

---

## SAS-Bench data (not tracked)
Available here: `https://huggingface.co/datasets/aleversn/SAS-Bench`
Place the downloaded full files at: data/
This folder is intentionally not committed to Git.


---

## Pipeline
### Data preperation
`python -m scripts.SAS_Math_prep.py`
- Filter the data

`python -m scripts.prep_SAS-Bench_multi_samples`
- Shuffle and split the exemplar & test subset

### Rubric generation
`python -m scripts.initial_rubric_gen_multi_samples`
- For initial rubric generation

`python -m scripts.rate_rubrics`
- Self-evaluation of the rubrics

`python -m scripts.rubric_refinement_by_feedback`
- Use the result from scripts/rate_rubrics.py to refine the initial rubrics

## Grading (command examples, adjust for other conditions)
`python -m src/grader_SAS \
  --dataset data.SAS-Bench.math_test_student.jsonl \
  --rubric  rubrics/math_refined_rubrics_multi.jsonl \
  --model o3-mini \
  --persona strict \
  --prompt_style one_shot`
- Grading with rubrics

`python -m  src.grader_no_rubric \
  --dataset data/SAS-Bench/math_test_student.jsonl \
  --model o3-mini \
  --persona neutral \
  --prompt_style zero_shot`
- Grading without rubrics

## Evaluation
`python -m src.evaluations.evaluation_in_one_go`
- Batch (all experiments) evaluations -> percent-of-maximum

`python -m src.evaluations.evaluate \
  results/predictions/<your_run>.csv`
- Raw score single run evaluation
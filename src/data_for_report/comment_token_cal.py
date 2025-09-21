from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

"""
To recalculate the output token difference between two groups of CSV files
"""

load_dotenv()
client = OpenAI()

# Configuration
group1_csvs = [ # original
    "results/predictions/o3mini_grader_20250831_205437_3b0e67.csv",
    "results/predictions/o3mini_grader_20250831_221436_aa9760.csv",
    "results/predictions/o3mini_grader_20250831_223605_ed5cab.csv",
    "results/predictions/o3mini_grader_20250831_225709_210cec.csv",
    "results/predictions/o3mini_grader_20250831_231931_f1fc75.csv",
]

group2_csvs = [ # no analysis
    "results/predictions/o3_grader_20250913_185734_7936be.csv",
    "results/predictions/o3_grader_20250913_183617_dcfb46.csv",
    "results/predictions/o3_grader_20250913_181439_0bedeb.csv",
    "results/predictions/o3_grader_20250913_175320_7ffa91.csv",
    "results/predictions/o3_grader_20250913_173146_dbe5ce.csv",
]

def get_token_count(text: str, model="o3-mini") -> int:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            temperature=1,
            max_completion_tokens=10
        )
        return response.usage.prompt_tokens
    except Exception as e:
        print(f"Error getting token count: {e}")
        return 0

def sum_tokens_from_csvs(csv_paths: list[str], column: str = "comments") -> int:
    total_tokens = 0
    for path in csv_paths:
        df = pd.read_csv(path)
        if column not in df.columns:
            print(f"Column '{column}' not found in {path}, skipping...")
            continue
        for text in df[column].dropna().astype(str):
            total_tokens += get_token_count(text)
    return total_tokens


print("Group 1...")
tokens_group1 = sum_tokens_from_csvs(group1_csvs)

print("Group 2...")
tokens_group2 = sum_tokens_from_csvs(group2_csvs)

# Results
raw_diff = tokens_group2 - tokens_group1
pct_diff = (raw_diff / tokens_group1) * 100 if tokens_group1 > 0 else float('inf')


print(f"Group 1 total tokens: {tokens_group1}")
print(f"Group 2 total tokens: {tokens_group2}")
print(f"Raw difference: {raw_diff}")
print(f"Percentage difference: {pct_diff:.2f}%")

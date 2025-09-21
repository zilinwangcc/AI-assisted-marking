import json
import pandas as pd

# Load the JSONL file
file_path = "data/SAS-Bench/datasets/7_Math_ShortAns.jsonl"
with open(file_path, 'r') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

# Convert to DataFrame
df = pd.DataFrame(data)

# Filter questions with at least 4 samples
grouped = df.groupby('question')
filtered_groups = [group.head(8) for _, group in grouped if len(group) >= 4]

# reassign IDs
new_data = []
new_id = 0
for group in filtered_groups:
    for _, row in group.iterrows():
        new_entry = row.copy()
        new_entry['id'] = f'Math_ShortAns_{new_id}'
        new_data.append(new_entry.to_dict())
        new_id += 1

# Save to a new JSONL file
output_path = "data/SAS-Bench/cleaned_Math_ShortAns.jsonl"
with open(output_path, 'w') as f:
    for entry in new_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

input_path = "training_dataset/0727pickplace/first100/meta/modified_tasks.jsonl"
output_path = "training_dataset/0727pickplace/first100/meta/modified_tasks_truncated.jsonl"

MAX_LINES = 20641

with open(input_path, "r") as infile:
    lines = [next(infile) for _ in range(MAX_LINES)]

with open(output_path, "w") as outfile:
    outfile.writelines(lines)

print(f"保留了前 {MAX_LINES} 行，保存到: {output_path}")

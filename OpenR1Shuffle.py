from datasets import load_dataset
import random
import json

# Load the dataset (adjust the split name if necessary)
dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")

# Select a single random entry from the dataset
index = random.sample(range(len(dataset)), 1)[0]
entry = dataset[index]

# Extract the question from the entry - try 'instruction' key, otherwise use 'problem'
question = entry.get("instruction", entry.get("problem", "Question not available"))

# Extract the answer from the entry - assume 'output' key and extract text after the last '=' if possible
output_field = entry.get("output", None)
if output_field:
    answer_start_index = output_field.rfind("=")
    if answer_start_index != -1:
        answer = output_field[answer_start_index+1:].strip()
    else:
        answer = output_field.strip()
else:
    answer = "Answer not available"

# Create a new JSON object
data = {"question": question, "answer": answer}

# Append the JSON object as a new line to a JSONL file
with open("random_R1.jsonl", "a") as f:
    json.dump(data, f)
    f.write("\n")

print(f"Random problem appended to random_R1.jsonl: {data}")
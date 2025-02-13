from datasets import load_dataset
import random
import json

# Load the dataset using the default configuration (login using e.g. `huggingface-cli login` may be required)
dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")

# Select a single random entry from the dataset
index = random.sample(range(len(dataset)), 1)[0]
entry = dataset[index]

# Extract the question and answer from the entry
question = entry.get("problem", "Question not available")
answer = entry.get("answer", "Answer not available")

# Create a new JSON object with the question and answer
data = {"question": question, "answer": answer}

# Append the JSON object as a new line to a JSONL file
with open("random_R1.jsonl", "a") as f:
    json.dump(data, f)
    f.write("\n")

print(f"Random problem appended to random_R1.jsonl: {data}")
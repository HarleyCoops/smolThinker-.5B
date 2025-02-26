# Open-R1-Math-220k Fine-tuning Project

## Project Overview

This project aims to fine-tune a small language model (based on Qwen 2.5 .5B distilled) on the Open-R1-Math-220k dataset for mathematical reasoning. We're building on previous work that used GSM8K, now leveraging a much larger synthetic dataset (220k examples vs. ~8k) to test how well small models can retain knowledge and reason through complex problems.

## Key Idea

The Open-R1-Math-220k dataset is a high-quality, synthetic dataset designed for training reasoning models. It's significantly larger than GSM8K (220k vs. ~8k) and has been filtered to maintain a high level of correctness and clarity. This makes it an excellent candidate for continued fine-tuning of our existing model.

We're not starting from scratch; we're building on the existing model's knowledge which started as the Qwen 2.5 .5B distilled version. This project tests what models retain from one training to the next and how small of a model can be effectively trained to reason through complex problem solving.

## Implementation Steps

### 1. Dataset Loading and Exploration

```python
from datasets import load_dataset

# Load the dataset
openr1_dataset = load_dataset("open-r1-math-220k")

# Inspect structure
print(openr1_dataset)
print(openr1_dataset['train'][0])  # Examine a sample

# Check dataset size
print(f"Training examples: {len(openr1_dataset['train'])}")
if 'validation' in openr1_dataset:
    print(f"Validation examples: {len(openr1_dataset['validation'])}")
```

**Implementation Notes:**
- Verify that the dataset has the expected structure with "instruction" and "output" fields
- Examine several examples to understand the format of questions and answers
- Check if there's a validation split; if not, you'll need to create one

### 2. Data Preprocessing

```python
def get_openr1_questions(dataset, split="train"):
    questions = []
    answers = []
    reasonings = []

    for example in dataset[split]:
        question = example["instruction"]
        output = example["output"]
        
        # Extract answer from the end of the output
        answer_start_index = output.rfind("=") + 1
        reasoning = output[:answer_start_index]
        answer = output[answer_start_index:].strip()
        
        questions.append(question)
        answers.append(answer)
        reasonings.append(reasoning)
        
    return questions, answers, reasonings

# Process training data
train_questions, train_answers, train_reasonings = get_openr1_questions(openr1_dataset, "train")

# Process validation data (if available)
if 'validation' in openr1_dataset:
    val_questions, val_answers, val_reasonings = get_openr1_questions(openr1_dataset, "validation")
else:
    # Create a validation split if not available
    from sklearn.model_selection import train_test_split
    train_questions, val_questions, train_answers, val_answers, train_reasonings, val_reasonings = train_test_split(
        train_questions, train_answers, train_reasonings, test_size=0.1, random_state=42
    )
```

**Implementation Notes:**
- The function extracts questions, answers, and reasoning steps from the dataset
- Adjust the answer extraction logic based on the actual format of the dataset
- If no validation split exists, create one using train_test_split

### 3. Answer Extraction and Reward Functions

```python
def extract_openr1_answer(generated_text):
    answer_start_index = generated_text.rfind("=") + 1
    return generated_text[answer_start_index:].strip()

def correctness_reward_func(generated_texts, reference_answers):
    rewards = []
    for gen_text, ref_answer in zip(generated_texts, reference_answers):
        extracted_answer = extract_openr1_answer(gen_text)
        try:
            if abs(float(extracted_answer) - float(ref_answer)) < 1e-8:
                rewards.append(1)
            else:
                rewards.append(0)
        except (ValueError, TypeError):
            print("Error comparing:", extracted_answer, ref_answer, gen_text)
            rewards.append(0)
    return rewards

# Test the extraction and reward functions on a few examples
test_outputs = [
    "2 + 2 = 4",
    "The answer is 10.5",
    "After calculating, we get x = 42"
]
test_answers = ["4", "10.5", "42"]

for output, answer in zip(test_outputs, test_answers):
    extracted = extract_openr1_answer(output)
    print(f"Output: {output}")
    print(f"Extracted: {extracted}")
    print(f"Reference: {answer}")
    print(f"Match: {extracted == answer}\n")
```

**Implementation Notes:**
- The extraction function needs to be adapted based on the actual format of answers in the dataset
- Test the extraction function on various examples to ensure it works correctly
- The reward function compares extracted answers with reference answers, with a small tolerance for floating-point comparisons

### 4. Training Setup

```python
from transformers import TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # Start with one epoch
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Adjust based on GPU memory
    learning_rate=1e-6,  # Lower learning rate for continued fine-tuning
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Set up the trainer
trainer = GRPOTrainer(
    model,
    args=training_args,
    train_dataset=train_questions,
    eval_dataset=val_questions,
    tokenizer=tokenizer,
    compute_metrics=None,
    grpo_loss_kwargs={
        "answer_grounding_loss_weight": answer_grounding_loss_weight,
    }
)
```

**Implementation Notes:**
- Start with a lower learning rate (1e-6) since we're continuing fine-tuning
- Keep batch size at 1 but adjust gradient accumulation steps as needed
- Set up regular evaluation to monitor for overfitting
- You may need to adjust the GRPO loss kwargs based on your specific implementation

### 5. Training and Evaluation

```python
# Start training
trainer.train()

# Save the model
trainer.save_model("./openr1_finetuned_model")

# Evaluate on test examples
def evaluate_model(model, tokenizer, questions, answers, num_examples=10):
    correct = 0
    for i in range(min(num_examples, len(questions))):
        inputs = tokenizer(questions[i], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=512)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        extracted_answer = extract_openr1_answer(generated_text)
        try:
            if abs(float(extracted_answer) - float(answers[i])) < 1e-8:
                correct += 1
        except (ValueError, TypeError):
            pass
            
        print(f"Question: {questions[i]}")
        print(f"Generated: {generated_text}")
        print(f"Extracted answer: {extracted_answer}")
        print(f"Reference answer: {answers[i]}")
        print(f"Correct: {extracted_answer == answers[i]}\n")
    
    print(f"Accuracy: {correct/num_examples:.2f}")

# Evaluate on validation set
evaluate_model(model, tokenizer, val_questions, val_answers, num_examples=20)
```

**Implementation Notes:**
- Monitor training loss and rewards during training
- Save the best model based on validation performance
- Implement a simple evaluation function to test the model on examples
- Consider implementing more sophisticated evaluation metrics if needed

## Additional Considerations

1. **Dataset Size Management**: The full dataset is large (220k examples). Consider starting with a subset for initial testing.

2. **Hyperparameter Tuning**: Experiment with different learning rates, training epochs, and reward weights.

3. **Format Consistency**: If the Open-R1 dataset has consistent formatting, you may be able to simplify or remove some format-related rewards.

4. **Memory Management**: Be mindful of GPU memory usage, especially with a large dataset.

5. **Incremental Testing**: Test each component thoroughly before running the full training pipeline.

## Getting Started

1. Clone this repository
2. Install the required dependencies
3. Download the Open-R1-Math-220k dataset
4. Run the dataset exploration code to understand its structure
5. Implement and test the preprocessing and reward functions
6. Run the training pipeline with a small subset of data
7. Scale up to the full dataset once everything is working correctly

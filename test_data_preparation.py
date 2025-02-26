"""
Test script for data preparation function.

This script tests the fixed data preparation function to ensure input and label shapes match.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from openr1_finetuning import prepare_training_data, format_prompt

def main():
    """Main function to test data preparation."""
    print("=" * 50)
    print("Testing Data Preparation Function")
    print("=" * 50)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Load a few examples from the dataset
    print("\nLoading dataset...")
    ds = load_dataset("open-r1/OpenR1-Math-220k")
    examples = ds['train'].select(range(3))  # Get 3 examples
    
    # Extract problems, solutions, and answers
    problems = [example['problem'] for example in examples]
    solutions = [example['solution'] for example in examples]
    answers = [example['answer'] for example in examples]
    
    # Prepare training data
    print("\nPreparing training data...")
    training_data = prepare_training_data(problems, solutions, answers, tokenizer)
    
    # Check shapes
    print("\nChecking tensor shapes:")
    for i, data in enumerate(training_data):
        print(f"\nExample {i+1}:")
        print(f"Input IDs shape: {data['input_ids'].shape}")
        print(f"Attention mask shape: {data['attention_mask'].shape}")
        print(f"Labels shape: {data['labels'].shape}")
        
        # Verify shapes match
        if data['input_ids'].shape == data['labels'].shape:
            print("✓ Input and label shapes match")
        else:
            print("✗ Input and label shapes don't match")
        
        # Check that prompt tokens are masked in labels
        prompt = format_prompt(problems[i])
        prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt_length = len(prompt_tokens)
        
        # Check first few labels are -100
        if (data['labels'][:prompt_length] == -100).all():
            print("✓ Prompt tokens are correctly masked in labels")
        else:
            print("✗ Prompt tokens are not correctly masked in labels")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 
"""
Test script for training data preparation.

This script tests the training data preparation functions with the actual Open-R1-Math-220k dataset.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

def format_prompt(problem):
    """
    Format the prompt for the model.
    
    Args:
        problem: The mathematical problem to format
        
    Returns:
        The formatted prompt
    """
    return f"""Solve the following mathematical problem step by step. After solving, clearly state the final answer.

Problem: {problem}

Solution:"""

def prepare_training_data_sample(problems, solutions, answers, tokenizer):
    """
    Prepare a sample of training data for the model.
    
    Args:
        problems: List of problems
        solutions: List of solutions
        answers: List of answers
        tokenizer: The tokenizer
        
    Returns:
        The prepared training data
    """
    training_data = []
    
    for problem, solution, answer in zip(problems, solutions, answers):
        prompt = format_prompt(problem)
        
        # Format the full solution with a clear answer at the end
        if "The answer is" not in solution and "the answer is" not in solution.lower():
            full_solution = solution + f"\n\nThe answer is: {answer}"
        else:
            full_solution = solution
        
        training_data.append({
            "prompt": prompt,
            "full_solution": full_solution
        })
    
    return training_data

def main():
    """Main function to test training data preparation."""
    print("=" * 50)
    print("Testing Training Data Preparation")
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
    training_data = prepare_training_data_sample(problems, solutions, answers, tokenizer)
    
    # Print the prepared training data
    print("\nPrepared training data:")
    for i, data in enumerate(training_data):
        print(f"\nExample {i+1}:")
        print(f"Prompt:\n{data['prompt']}")
        print(f"\nFull solution (last 100 chars):\n...{data['full_solution'][-100:]}")
        
        # Check if the answer is clearly stated
        if "The answer is" in data['full_solution'] or "the answer is" in data['full_solution'].lower():
            print("\nAnswer is clearly stated ✓")
        else:
            print("\nAnswer is NOT clearly stated ✗")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 
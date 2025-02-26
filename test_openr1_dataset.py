"""
Test script for Open-R1-Math-220k dataset loading and preprocessing.

This script tests the dataset loading and preprocessing functions with the actual dataset structure.
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re

def load_and_explore_dataset():
    """Load and explore the Open-R1-Math-220k dataset."""
    print("Loading Open-R1-Math-220k dataset...")
    openr1_dataset = load_dataset("open-r1/OpenR1-Math-220k")
    
    # Inspect structure
    print("\nDataset structure:")
    print(openr1_dataset)
    
    # Examine a sample
    print("\nSample example:")
    print(openr1_dataset['train'][0])
    
    # Check dataset size
    print(f"\nTraining examples: {len(openr1_dataset['train'])}")
    if 'validation' in openr1_dataset:
        print(f"Validation examples: {len(openr1_dataset['validation'])}")
    else:
        print("No validation split found in the dataset.")
    
    return openr1_dataset

def preprocess_dataset(dataset, split="train"):
    """
    Extract problems, solutions, and answers from the dataset.
    
    Args:
        dataset: The Open-R1-Math-220k dataset
        split: The dataset split to process
        
    Returns:
        Tuple of (problems, solutions, answers)
    """
    problems = []
    solutions = []
    answers = []
    
    print(f"Processing {split} split...")
    
    for example in dataset[split]:
        problem = example["problem"]
        solution = example["solution"]
        answer = example["answer"]
        
        problems.append(problem)
        solutions.append(solution)
        answers.append(answer)
    
    print(f"Processed {len(problems)} examples from {split} split.")
    return problems, solutions, answers

def create_train_val_split(dataset):
    """
    Create training and validation splits from the dataset.
    
    Args:
        dataset: The Open-R1-Math-220k dataset
        
    Returns:
        Tuple of (train_problems, train_solutions, train_answers, 
                 val_problems, val_solutions, val_answers)
    """
    # Process training data
    train_problems, train_solutions, train_answers = preprocess_dataset(dataset, "train")
    
    # Process validation data (if available)
    if 'validation' in dataset:
        val_problems, val_solutions, val_answers = preprocess_dataset(dataset, "validation")
    else:
        # Create a validation split if not available
        print("Creating validation split from training data...")
        train_problems, val_problems, train_solutions, val_solutions, train_answers, val_answers = train_test_split(
            train_problems, train_solutions, train_answers, test_size=0.1, random_state=42
        )
        print(f"Split into {len(train_problems)} training and {len(val_problems)} validation examples.")
    
    return (
        train_problems, train_solutions, train_answers,
        val_problems, val_solutions, val_answers
    )

def test_extraction_function():
    """Test the answer extraction function on sample outputs."""
    # Load a few examples from the dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k")
    examples = ds['train'].select(range(3))
    
    print("\nTesting answer extraction:")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Problem: {example['problem'][:100]}...")  # Show first 100 chars
        print(f"Expected answer: {example['answer']}")
        
        # In this dataset, we don't need to extract the answer as it's already provided
        # But we'll verify it's in the solution
        if example['answer'] in example['solution']:
            print(f"Answer found in solution: Yes")
        else:
            print(f"Answer found in solution: No")

def main():
    """Main function to test dataset loading and preprocessing."""
    print("=" * 50)
    print("Testing Open-R1-Math-220k Dataset Loading and Preprocessing")
    print("=" * 50)
    
    # Step 1: Load and explore dataset
    print("\n1. Loading and exploring dataset...")
    dataset = load_and_explore_dataset()
    
    # Step 2: Test preprocessing on a small subset
    print("\n2. Testing preprocessing on a small subset...")
    # Create a small subset for testing
    small_dataset = {
        'train': dataset['train'].select(range(min(100, len(dataset['train']))))
    }
    
    # Process the small subset
    problems, solutions, answers = preprocess_dataset(small_dataset, "train")
    
    # Print some examples
    print("\nSample processed examples:")
    for i in range(min(3, len(problems))):
        print(f"\nExample {i+1}:")
        print(f"Problem: {problems[i][:100]}...")  # Show first 100 chars of problem
        print(f"Solution: {solutions[i][:100]}...")  # Show first 100 chars of solution
        print(f"Answer: {answers[i]}")
    
    # Step 3: Test train/val split creation
    print("\n3. Testing train/val split creation...")
    train_p, train_s, train_a, val_p, val_s, val_a = create_train_val_split(small_dataset)
    
    print(f"\nTraining examples: {len(train_p)}")
    print(f"Validation examples: {len(val_p)}")
    
    # Step 4: Test answer extraction function
    print("\n4. Testing answer extraction function...")
    test_extraction_function()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 
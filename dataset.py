from datasets import load_dataset
import pandas as pd
import random
import argparse

def examine_dataset(dataset_config):
    print(f"\n{'='*50}")
    print(f"Examining OpenR1-Math-220k dataset with config: {dataset_config}")
    print(f"{'='*50}\n")
    
    # Load the dataset
    try:
        ds = load_dataset("open-r1/OpenR1-Math-220k", dataset_config)
        
        # 1) Print dataset overview
        print("Dataset Overview:")
        print(ds)
        print("\n")
        
        # 2) Show number of examples for each split
        for split in ds.keys():
            print(f"Number of {split} examples:", len(ds[split]))
        print("\n")
        
        # 3) Inspect dataset columns and types
        print("Column names:", ds['train'].column_names)
        print("Features:", ds['train'].features)
        print("\n")
        
        # 4) Peek at first example
        print("First example from train split:")
        print(ds['train'][0])
        print("\n")
        
        # 5) Create sample DataFrame
        print("Sample DataFrame (first 5 examples):")
        sample_df = pd.DataFrame(ds['train'][:5])
        print(sample_df)
        print("\n")
        
    except Exception as e:
        print(f"Error loading dataset with config '{dataset_config}': {str(e)}")

def get_random_question(split="train", config="default"):
    """
    Get a random question from the OpenR1-Math-220k dataset.
    
    Args:
        split (str): Dataset split to use ('train' or 'test')
        config (str): Dataset configuration ('default', 'all', or 'extended')
    
    Returns:
        dict: A dictionary containing the question and its solution
    """
    try:
        # Load the dataset
        ds = load_dataset("open-r1/OpenR1-Math-220k", config)
        
        # Get a random index
        random_idx = random.randint(0, len(ds[split]) - 1)
        
        # Get the random example
        example = ds[split][random_idx]
        
        # Print available fields for debugging
        print("\nAvailable fields in the dataset:", example.keys())
        
        print("\n=== Random Math Question ===")
        # Try different possible field names for the question
        question = example.get('question', example.get('input', example.get('problem', None)))
        solution = example.get('solution', example.get('output', example.get('answer', None)))
        
        if question is not None:
            print(f"Question: {question}")
        else:
            print("Warning: Could not find question field. Raw example:", example)
            
        if solution is not None:
            print(f"\nSolution: {solution}")
        else:
            print("\nWarning: Could not find solution field. Raw example:", example)
        
        return example
        
    except Exception as e:
        print(f"Error getting random question: {str(e)}")
        print("Full error details:", e)
        return None

def main():
    parser = argparse.ArgumentParser(description='Dataset examination and random question generator')
    parser.add_argument('--function', choices=['examine', 'random'], default='examine',
                      help='Which function to run: examine (dataset examination) or random (get random question)')
    parser.add_argument('--split', default='train', choices=['train', 'test'],
                      help='Dataset split to use (for random question)')
    parser.add_argument('--config', default='default', choices=['default', 'all', 'extended'],
                      help='Dataset configuration to use')
    
    args = parser.parse_args()
    
    if args.function == 'random':
        get_random_question(split=args.split, config=args.config)
    else:
        configs = ["default", "all", "extended"]
        for config in configs:
            examine_dataset(config)

if __name__ == "__main__":
    main()

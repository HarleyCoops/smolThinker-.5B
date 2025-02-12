from datasets import load_dataset
import pandas as pd

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

def main():
    configs = ["default", "all", "extended"]
    for config in configs:
        examine_dataset(config)

if __name__ == "__main__":
    main()

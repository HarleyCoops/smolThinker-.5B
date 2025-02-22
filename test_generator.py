from reward_function_generator import RewardFunctionGenerator
from datasets import load_dataset
import importlib.util
import sys
from pathlib import Path

def test_reward_functions(dataset_name: str, config: str = None):
    """Test reward functions for a dataset"""
    print(f"\nGenerating reward functions for dataset: {dataset_name}")
    
    # Generate reward functions
    generator = RewardFunctionGenerator(dataset_name, config)
    rewards_path = generator.generate_reward_functions()
    analysis_path = generator.save_analysis()
    
    print(f"\nReward functions saved to: {rewards_path}")
    print(f"Dataset analysis saved to: {analysis_path}")
    
    # Import the generated reward functions
    module_name = Path(rewards_path).stem
    spec = importlib.util.spec_from_file_location(module_name, rewards_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Get a sample from the dataset
    ds = load_dataset(dataset_name, config)
    sample = ds['train'][0]
    
    # Create a test completion
    if "answers" in sample and isinstance(sample["answers"], dict) and "text" in sample["answers"]:
        answer = sample["answers"]["text"][0]  # SQuAD format
    else:
        answer = sample.get("answer", "4")  # Other formats
    
    completion = [[{
        "role": "assistant",
        "content": f"""<reasoning>
Let's solve this step by step:
1. First, we need to understand the question
2. Then, we can determine the approach
3. Finally, we calculate the answer
</reasoning>
<answer>{answer}</answer>"""
    }]]
    
    # Test each reward function
    print("\nTesting reward functions:")
    for func_name, weight in module.REWARD_WEIGHTS.items():
        reward_func = getattr(module, func_name)
        if func_name == 'correctness_reward':
            result = reward_func(None, completion, sample)
        else:
            result = reward_func(completion)
        print(f"\n{func_name} (weight: {weight}):")
        print(f"Score: {result}")

if __name__ == "__main__":
    # Test with OpenR1-Math dataset
    test_reward_functions("open-r1/OpenR1-Math-220k")
    
    # Test with SQuAD dataset
    test_reward_functions("squad")

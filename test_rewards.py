from datasets import load_dataset
from abstraction import build_reward_functions_from_dataset
from reward_functions import load_reward_functions

def test_dataset_rewards(dataset_name: str, config: str = None):
    """Test reward function generation and saving for a dataset"""
    print(f"\nTesting rewards for dataset: {dataset_name}")
    
    # Build and save reward functions
    reward_funcs, save_path = build_reward_functions_from_dataset(dataset_name, config)
    print(f"Reward functions saved to: {save_path}")
    
    # Load the saved reward functions metadata
    metadata = load_reward_functions(dataset_name)
    print("\nReward Functions Metadata:")
    for name, info in metadata["reward_functions"].items():
        print(f"\n{name}:")
        print(f"  Type: {info['type']}")
        print(f"  Description: {info['description']}")
    
    # Test the reward functions
    ds = load_dataset(dataset_name, config)
    sample = ds['train'][0]
    
    # Create a test completion using the sample's answer
    if "answers" in sample and isinstance(sample["answers"], dict) and "text" in sample["answers"]:
        answer = sample["answers"]["text"][0]  # SQuAD format
    else:
        answer = sample.get("answer", "4")  # Other formats
    
    completion = [[{
        "role": "assistant",
        "content": f"<reasoning>Let's solve this step by step.</reasoning><answer>{answer}</answer>"
    }]]
    
    # Calculate rewards
    rewards = reward_funcs["correctness_reward"](None, completion, sample)
    print(f"\nSample reward calculation: {rewards}")

if __name__ == "__main__":
    # Test with OpenR1-Math dataset
    test_dataset_rewards("open-r1/OpenR1-Math-220k")
    
    # Test with SQuAD dataset
    test_dataset_rewards("squad")

import numpy as np
from pathlib import Path

def test_reward_functions(dataset_name: str):
    """Test the reward functions for a dataset"""
    
    print("\nTesting reward functions:")
    
    # Import the reward functions
    safe_name = dataset_name.replace('/', '_').replace('-', '_')
    module_path = Path(f"generated_rewards/{safe_name}_rewards.py")
    
    # Create a sample completion for testing
    completion = [{
        "content": """<reasoning>
1. First, let's understand what we're looking for
2. Then, let's solve step by step
3. Finally, we can calculate the result
x = 5 + 3 = 8
</reasoning>
<answer>8</answer>"""
    }]
    
    # Create a sample data point
    sample = {
        "answers": {"text": ["8"]},
        "answer": "8"
    }
    
    # Test each reward function
    with open(module_path, 'r') as f:
        exec(f.read(), globals())
    
    # Test correctness reward
    result = correctness_reward(None, [completion], sample)
    print(f"\ncorrectness_reward (weight: {REWARD_WEIGHTS['correctness_reward']}):")
    print(f"Score: {np.array(result)}")
    
    # Test format reward
    result = format_reward(completions=[completion])
    print(f"\nformat_reward (weight: {REWARD_WEIGHTS['format_reward']}):")
    print(f"Score: {np.array(result)}")
    
    # Test reasoning quality reward
    result = reasoning_quality_reward(completions=[completion])
    print(f"\nreasoning_quality_reward (weight: {REWARD_WEIGHTS['reasoning_quality_reward']}):")
    print(f"Score: {np.array(result)}")
    
    # Test step clarity reward
    result = step_clarity_reward(completions=[completion])
    print(f"\nstep_clarity_reward (weight: {REWARD_WEIGHTS['step_clarity_reward']}):")
    print(f"Score: {np.array(result)}")

if __name__ == "__main__":
    # Test with OpenR1-Math dataset
    print("\nGenerating reward functions for dataset: open-r1/OpenR1-Math-220k")
    test_reward_functions("open-r1/OpenR1-Math-220k")
    
    # Test with SQuAD dataset
    print("\nGenerating reward functions for dataset: squad")
    test_reward_functions("squad")

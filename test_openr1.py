from datasets import load_dataset
from abstraction import build_reward_functions_from_dataset

# Build reward functions for OpenR1-Math dataset
reward_funcs = build_reward_functions_from_dataset('open-r1/OpenR1-Math-220k')
print('\nAvailable reward functions:', list(reward_funcs.keys()))

# Get a sample from the dataset
ds = load_dataset('open-r1/OpenR1-Math-220k')
sample = ds['train'][0]

# Test the reward function with a completion
completions = [[{
    'role': 'assistant',
    'content': f'<reasoning>Let\'s solve this step by step:\n1. Given v_R = 4 km/h and v_B = 10 km/h\n2. These values satisfy the equations</reasoning><answer>{sample["answer"]}</answer>'
}]]

print('\nExample reward calculation:')
print(reward_funcs['correctness_reward'](None, completions, sample))

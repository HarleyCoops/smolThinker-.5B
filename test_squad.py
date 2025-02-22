from datasets import load_dataset
from abstraction import build_reward_functions_from_dataset

# Build reward functions for SQuAD dataset
reward_funcs = build_reward_functions_from_dataset('squad')
print('\nAvailable reward functions:', list(reward_funcs.keys()))

# Get a sample from the dataset
ds = load_dataset('squad')
sample = ds['train'][0]

# Test the reward function with a completion
completions = [[{
    'role': 'assistant',
    'content': f'<reasoning>Based on the context, the Virgin Mary appeared to Saint Bernadette Soubirous in Lourdes, France in 1858.</reasoning><answer>{sample["answers"]["text"][0]}</answer>'
}]]

print('\nExample reward calculation:')
print(reward_funcs['correctness_reward'](None, completions, sample))

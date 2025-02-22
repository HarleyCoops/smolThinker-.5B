import re
import json
from typing import Any, Dict, List, Callable
from pathlib import Path

def save_reward_functions(dataset_name: str, reward_funcs: Dict[str, Callable], save_dir: str = "reward_functions") -> str:
    """
    Save reward functions configuration and metadata for a dataset.
    
    Args:
        dataset_name: Name of the dataset these reward functions are for
        reward_funcs: Dictionary of reward functions
        save_dir: Directory to save the reward functions in
        
    Returns:
        Path to the saved reward functions file
    """
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a safe filename from dataset name
    safe_name = dataset_name.replace('/', '_').replace('-', '_')
    save_path = Path(save_dir) / f"{safe_name}_rewards.json"
    
    # Extract reward function metadata
    reward_metadata = {
        "dataset_name": dataset_name,
        "reward_functions": {}
    }
    
    for name, func in reward_funcs.items():
        reward_metadata["reward_functions"][name] = {
            "name": name,
            "description": func.__doc__ or "No description available",
            "type": "correctness" if "correctness" in name else 
                    "format" if "format" in name else 
                    "reasoning" if "reasoning" in name else "other"
        }
    
    # Save metadata
    with open(save_path, 'w') as f:
        json.dump(reward_metadata, f, indent=2)
    
    return str(save_path)

def load_reward_functions(dataset_name: str, save_dir: str = "reward_functions") -> Dict[str, Any]:
    """
    Load reward functions configuration for a dataset.
    
    Args:
        dataset_name: Name of the dataset to load reward functions for
        save_dir: Directory where reward functions are saved
        
    Returns:
        Dictionary containing reward function metadata
    """
    safe_name = dataset_name.replace('/', '_').replace('-', '_')
    load_path = Path(save_dir) / f"{safe_name}_rewards.json"
    
    if not load_path.exists():
        raise FileNotFoundError(f"No reward functions found for dataset {dataset_name}")
    
    with open(load_path) as f:
        return json.load(f)

# Helper functions that will be used by the reward functions
def extract_xml_answer(text: str) -> str:
    """Helper to extract answer from XML tags"""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_xml_reasoning(text: str) -> str:
    """Helper to extract reasoning from XML tags"""
    match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_answer_text(answer_data) -> str:
    """Helper to extract answer text from potentially nested structures"""
    if isinstance(answer_data, dict):
        if 'text' in answer_data:
            text_data = answer_data['text']
            if isinstance(text_data, list):
                return str(text_data[0]) if text_data else ""
            return str(text_data)
    return str(answer_data) if answer_data is not None else ""

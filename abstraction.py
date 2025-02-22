import re
from typing import Any, Dict, List, Callable, Tuple
from datasets import load_dataset
from reward_functions import save_reward_functions, extract_answer_text, extract_xml_answer, extract_xml_reasoning

def analyze_dataset_schema(dataset_name: str, config: str = None) -> Dict[str, Any]:
    """
    Inspects any dataset's structure and returns a comprehensive analysis.
    
    Args:
        dataset_name: The HuggingFace dataset identifier (e.g., "open-r1/OpenR1-Math-220k")
        config: Dataset configuration name
        
    Returns:
        Dict containing dataset analysis including:
        - Column names and types
        - Detected patterns
        - Special format requirements
        - Answer type information
    """
    try:
        # Load dataset
        if config:
            ds = load_dataset(dataset_name, config)
        else:
            # Try loading without config first
            try:
                ds = load_dataset(dataset_name)
            except ValueError as e:
                # If that fails, try with 'plain_text' config
                if "BuilderConfig" in str(e):
                    ds = load_dataset(dataset_name, 'plain_text')
                else:
                    raise e
        
        # Get training split for analysis
        train_data = ds['train']
        
        # Get first example to analyze structure
        first_example = train_data[0]
        
        # Print debug info
        print("\nDataset First Example:")
        print(first_example)
        
        # Analyze structure
        columns = list(first_example.keys()) if isinstance(first_example, dict) else []
        print("\nDetected columns:", columns)
        
        # Initialize analysis dict
        analysis = {
            "columns": columns,
            "features": {},
            "patterns": {},
            "answer_format": None,
            "special_requirements": []
        }
        
        # Detect question/answer columns
        for col in columns:
            col_lower = col.lower()
            if any(q in col_lower for q in ['question', 'problem', 'input']):
                analysis["question_column"] = col
            elif any(a in col_lower for a in ['answer', 'solution', 'output']):
                analysis["answer_column"] = col
        
        # Sample data analysis
        sample_size = min(100, len(train_data))
        samples = train_data[:sample_size]
        
        # Analyze answer patterns
        if "answer_column" in analysis:
            answer_col = analysis["answer_column"]
            answers = []
            def extract_answer_text(answer_data) -> str:
                """Helper to extract answer text from potentially nested structures"""
                if isinstance(answer_data, dict):
                    if 'text' in answer_data:
                        text_data = answer_data['text']
                        if isinstance(text_data, list):
                            return str(text_data[0]) if text_data else ""
                        return str(text_data)
                return str(answer_data) if answer_data is not None else ""
            
            # Process samples
            for sample in samples:
                try:
                    answer = sample[answer_col]
                    answer_text = extract_answer_text(answer)
                    answers.append(answer_text)
                except (KeyError, TypeError, IndexError):
                    continue  # Skip problematic samples silently
            
            # Detect numeric answers
            numeric_count = sum(1 for a in answers if re.match(r'^\s*-?\d+\.?\d*\s*$', a))
            if numeric_count / sample_size > 0.5:
                analysis["patterns"]["numeric_answers"] = True
                analysis["answer_format"] = "numeric"
            
            # Detect XML format
            xml_count = sum(1 for a in answers if '<' in a and '>' in a)
            if xml_count / sample_size > 0.3:
                analysis["patterns"]["xml_format"] = True
                analysis["special_requirements"].append("xml_tags")
            
            # Check for multi-step solutions
            step_count = sum(1 for a in answers if '\n' in a or ';' in a)
            if step_count / sample_size > 0.3:
                analysis["patterns"]["multi_step"] = True
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        return {"error": str(e)}

def build_reward_functions_from_dataset(
    dataset_name: str,
    config: str = None,
    user_instructions: str = "",
    save_dir: str = "reward_functions"
) -> Tuple[Dict[str, Callable], str]:
    """
    Analyzes a dataset and builds appropriate reward functions.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        config: Dataset configuration
        user_instructions: Optional instructions for customizing rewards
        
    Returns:
        Dictionary of reward functions
    """
    # Analyze dataset
    analysis = analyze_dataset_schema(dataset_name, config)
    reward_funcs = {}
    
    def extract_xml_answer(text: str) -> str:
        """Helper to extract answer from XML tags"""
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def extract_xml_reasoning(text: str) -> str:
        """Helper to extract reasoning from XML tags"""
        match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    # Basic correctness reward
    if "answer_column" in analysis:
        def correctness_reward(prompts, completions, sample_data, **kwargs) -> List[float]:
            """Compare completion answer with ground truth"""
            scores = []
            for completion in completions:
                response = completion[0]["content"]
                extracted = extract_xml_answer(response)
                # Handle nested answer structures
                answer_data = sample_data[analysis["answer_column"]]
                if isinstance(answer_data, dict) and 'text' in answer_data:
                    if isinstance(answer_data['text'], list):
                        truth = str(answer_data['text'][0])
                    else:
                        truth = str(answer_data['text'])
                else:
                    truth = str(answer_data)
                
                # Handle numeric comparison if detected
                if analysis.get("patterns", {}).get("numeric_answers"):
                    try:
                        extracted_num = float(extracted)
                        truth_num = float(truth)
                        scores.append(2.0 if abs(extracted_num - truth_num) < 1e-6 else 0.0)
                    except ValueError:
                        scores.append(0.0)
                else:
                    # String comparison
                    scores.append(2.0 if extracted.strip() == truth.strip() else 0.0)
            return scores
        
        reward_funcs["correctness_reward"] = correctness_reward
    
    # Format reward for XML structure
    if analysis.get("patterns", {}).get("xml_format"):
        def format_reward(prompts, completions, **kwargs) -> List[float]:
            """Check XML format compliance"""
            scores = []
            for completion in completions:
                response = completion[0]["content"]
                has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))
                has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", response, re.DOTALL))
                
                # Score based on format compliance
                score = 0.0
                if has_answer:
                    score += 0.5
                if has_reasoning:
                    score += 0.5
                scores.append(score)
            return scores
            
        reward_funcs["format_reward"] = format_reward
    
    # Multi-step reasoning reward
    if analysis.get("patterns", {}).get("multi_step"):
        def reasoning_quality_reward(prompts, completions, **kwargs) -> List[float]:
            """Evaluate quality of reasoning steps"""
            scores = []
            for completion in completions:
                response = completion[0]["content"]
                reasoning = extract_xml_reasoning(response)
                
                # Count reasoning steps
                steps = [s.strip() for s in reasoning.split('\n') if s.strip()]
                
                # Score based on number of clear steps
                score = min(len(steps) * 0.25, 1.0)  # Cap at 1.0
                scores.append(score)
            return scores
            
        reward_funcs["reasoning_quality_reward"] = reasoning_quality_reward
    
    # Save the reward functions
    save_path = save_reward_functions(dataset_name, reward_funcs, save_dir)
    
    return reward_funcs, save_path

def combined_reward(prompts, completions, sample_data, reward_funcs_dict):
    """
    Aggregates multiple reward functions into a final score.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        sample_data: Ground truth data
        reward_funcs_dict: Dictionary of reward functions
        
    Returns:
        List of combined reward scores
    """
    total_rewards = [0.0] * len(completions)
    
    # Weight dictionary for different reward types
    weights = {
        "correctness_reward": 1.0,
        "format_reward": 0.5,
        "reasoning_quality_reward": 0.5
    }
    
    for name, func in reward_funcs_dict.items():
        try:
            partial_rewards = func(prompts=prompts, completions=completions, sample_data=sample_data)
            weight = weights.get(name, 1.0)
            total_rewards = [t + (p * weight) for t, p in zip(total_rewards, partial_rewards)]
        except Exception as e:
            print(f"Error in reward function {name}: {str(e)}")
            continue
    
    return total_rewards

# Example usage
if __name__ == "__main__":
    # Example dataset - OpenR1 Math
    dataset_name = "open-r1/OpenR1-Math-220k"
    config = None
    
    # Build and save reward functions
    reward_funcs, save_path = build_reward_functions_from_dataset(dataset_name, config)
    print(f"\nReward functions saved to: {save_path}")
    
    # Example data
    # Load dataset to get a real example
    ds = load_dataset(dataset_name, config)
    sample_data = ds['train'][0]
    
    print("\nSample data structure:")
    for key, value in sample_data.items():
        print(f"{key}: {value}")

    # Find question and answer columns
    question_col = None
    answer_col = None
    
    for col in sample_data.keys():
        if any(q in col.lower() for q in ['question', 'problem', 'input']):
            question_col = col
        elif any(a in col.lower() for a in ['answer', 'solution', 'output']):
            answer_col = col
    
    if question_col and answer_col:
        prompts = [[{"role": "system", "content": "You are a helpful math solver."},
                   {"role": "user", "content": str(sample_data[question_col])}]]
                   
        completions = [[{"role": "assistant", 
                        "content": f"<reasoning>\nLet's solve this step by step.\n</reasoning>\n<answer>{sample_data[answer_col]}</answer>"}]]
    else:
        print("\nWarning: Could not identify question/answer columns")
        prompts = [[{"role": "system", "content": "You are a helpful math solver."},
                   {"role": "user", "content": "2 + 2"}]]
        completions = [[{"role": "assistant", 
                        "content": "<reasoning>\nSimple addition\n</reasoning>\n<answer>4</answer>"}]]
    
    # Calculate rewards
    rewards = combined_reward(prompts, completions, sample_data, reward_funcs)
    print(f"\nReward functions generated for dataset: {dataset_name}")
    print(f"Available reward functions: {list(reward_funcs.keys())}")
    print(f"Sample reward calculation: {rewards}")

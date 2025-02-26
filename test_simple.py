from pathlib import Path

def generate_reward_functions(dataset_name: str):
    """Generate reward functions for a dataset"""
    
    # Generate reward functions
    output_dir = Path("generated_rewards")
    output_dir.mkdir(exist_ok=True)
    safe_name = dataset_name.replace('/', '_').replace('-', '_')
    output_path = output_dir / f"{safe_name}_rewards.py"
    
    # Write reward functions with proper formatting
    with open(output_path, 'w') as f:
        # Write imports
        f.write('import re\nfrom typing import List, Dict\n\n')
        
        # Write helper functions
        f.write('''def extract_xml_answer(text: str) -> str:
    """Extract answer from XML tags."""
    match = re.search(r"<answer>\\s*(.*?)\\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_xml_reasoning(text: str) -> str:
    """Extract reasoning from XML tags."""
    match = re.search(r"<reasoning>\\s*(.*?)\\s*</reasoning>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def correctness_reward(prompts, completions, sample_data, **kwargs) -> List[float]:
    """Compare completion answer with ground truth."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        extracted = extract_xml_answer(response)
        
        # Get ground truth answer
        answer_data = sample_data["answers" if "answers" in sample_data else "answer"]
        if isinstance(answer_data, dict) and "text" in answer_data:
            truth = str(answer_data["text"][0]) if isinstance(answer_data["text"], list) else str(answer_data["text"])
        else:
            truth = str(answer_data)
        
        # Compare answers
        scores.append(2.0 if extracted.strip() == truth.strip() else 0.0)
    return scores

def format_reward(completions, **kwargs) -> List[float]:
    """Check response format compliance."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", response, re.DOTALL))
        has_newlines = response.count("\\n") > 0
        proper_nesting = response.find("<reasoning>") < response.find("</reasoning>") < response.find("<answer>") < response.find("</answer>")
        
        score = 0.0
        if has_reasoning:
            score += 0.2
        if has_answer:
            score += 0.2
        if has_newlines:
            score += 0.05
        if proper_nesting:
            score += 0.05
        
        scores.append(score)
    return scores

def reasoning_quality_reward(completions, **kwargs) -> List[float]:
    """Evaluate quality of reasoning steps."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        reasoning = extract_xml_reasoning(response)
        steps = [s.strip() for s in reasoning.split("\\n") if s.strip()]
        
        score = 0.0
        if steps:
            # Base score for having steps
            score += 0.2
            
            # Bonus for number of steps (up to 0.3)
            score += min(len(steps) * 0.1, 0.3)
            
            # Check for mathematical notation
            if re.search(r"[=+\\-*/\\^]", reasoning):
                score += 0.2
            
            # Check for variable usage
            if re.search(r"[a-zA-Z]_?\\{?[a-zA-Z0-9]*\\}?", reasoning):
                score += 0.2
            
            # Cap at 1.0
            score = min(score, 1.0)
        
        scores.append(score)
    return scores

def step_clarity_reward(completions, **kwargs) -> List[float]:
    """Evaluate clarity and structure of solution steps."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        reasoning = extract_xml_reasoning(response)
        steps = reasoning.split("\\n")
        
        score = 0.0
        if steps:
            # Check for numbered/ordered steps
            if any(re.match(r"^[0-9]+[).] ", step.strip()) for step in steps):
                score += 0.2
            
            # Check for clear step progression
            if any(word in reasoning.lower() for word in ["first", "then", "next", "finally"]):
                score += 0.2
            
            # Check for mathematical clarity
            if re.search(r"=.*=", reasoning):
                score += 0.2
            
            # Check for explanatory text
            if re.search(r"[A-Za-z]{3,}", reasoning):
                score += 0.2
            
            # Penalize very long steps
            if max(len(step) for step in steps) > 200:
                score -= 0.1
        
        scores.append(max(0.0, score))
    return scores

# Reward function weights
REWARD_WEIGHTS = {
    "correctness_reward": 1.0,
    "format_reward": 0.5,
    "reasoning_quality_reward": 0.3,
    "step_clarity_reward": 0.2
}
'''.replace('\\s*', r'\s*')
   .replace('\\n', r'\n')
   .replace('\\{', r'\{')
   .replace('\\}', r'\}')
   .replace('\\^', r'\^')
   .replace('\\-', r'\-')
   .replace('Dict\n\n', 'Dict, \n\n')
   .replace('prompts completions', 'prompts, completions')
   .replace('completions **kwargs', 'completions, **kwargs')
   .replace('answer_data dict', 'answer_data, dict')
   .replace('answer_data["text"] list', 'answer_data["text"], list')
   .replace('</answer>" text', '</answer>", text')
   .replace('</reasoning>" text', '</reasoning>", text')
   .replace('</answer>" response', '</answer>", response')
   .replace('</reasoning>" response', '</reasoning>", response')
   .replace('\\^]" reasoning', '\\^]", reasoning')
   .replace('\\}?" reasoning', '\\}?", reasoning')
   .replace('=.*=" reasoning', '=.*=", reasoning')
   .replace('step.strip()', 'step.strip())')
   .replace('["first" "then"', '["first", "then",')
   .replace('"next" "finally"]', '"next", "finally"]')
   .replace('max(0.0 score', 'max(0.0, score')
   .replace('min(score 1.0', 'min(score, 1.0')
   .replace('min(len(steps) * 0.1 0.3', 'min(len(steps) * 0.1, 0.3')
   .replace('"correctness_reward": 1.0\n    "format_reward"', '"correctness_reward": 1.0,\n    "format_reward"')
   .replace('"format_reward": 0.5\n    "reasoning_quality_reward"', '"format_reward": 0.5,\n    "reasoning_quality_reward"')
   .replace('"reasoning_quality_reward": 0.3\n    "step_clarity_reward"', '"reasoning_quality_reward": 0.3,\n    "step_clarity_reward"'))
    
    print(f"\nReward functions saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test with OpenR1-Math dataset
    generate_reward_functions("open-r1/OpenR1-Math-220k")
    
    # Test with SQuAD dataset
    generate_reward_functions("squad")

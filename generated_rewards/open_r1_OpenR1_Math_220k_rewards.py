import re
from typing import List, Dict

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML tags."""
    match = re.search(r"<answer>[ 	]*(.*?)[ 	]*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_xml_reasoning(text: str) -> str:
    """Extract reasoning from XML tags."""
    match = re.search(r"<reasoning>[ 	]*(.*?)[ 	]*</reasoning>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def count_reasoning_steps(text: str) -> int:
    """Count the number of reasoning steps."""
    reasoning = extract_xml_reasoning(text)
    steps = [s.strip() for s in reasoning.split("\n") if s.strip()]
    return len(steps)

def check_xml_format(text: str) -> Dict[str, bool]:
    """Check XML format compliance."""
    return {
        "has_reasoning": bool(re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL)),
        "has_answer": bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL)),
        "has_newlines": text.count("\n") > 0,
        "proper_nesting": text.find("<reasoning>") < text.find("</reasoning>") < text.find("<answer>") < text.find("</answer>")
    }

def correctness_reward(prompts, completions, sample_data, **kwargs) -> List[float]:
    """Compare completion answer with ground truth."""
    scores = []
    for completion in completions:
        response = completion[0]['content']
        extracted = extract_xml_answer(response)
        
        # Get ground truth answer
        answer_data = sample_data['answer']
        if isinstance(answer_data, dict) and 'text' in answer_data:
            truth = str(answer_data['text'][0]) if isinstance(answer_data['text'], list) else str(answer_data['text'])
        else:
            truth = str(answer_data)
            
        # Compare answers
        
        try:
            extracted_num = float(extracted)
            truth_num = float(truth)
            scores.append(2.0 if abs(extracted_num - truth_num) < 1e-6 else 0.0)
        except ValueError:
            scores.append(2.0 if extracted.strip() == truth.strip() else 0.0)
        
            scores.append(2.0 if extracted.strip() == truth.strip() else 0.0)
    return scores

def format_reward(completions, **kwargs) -> List[float]:
    """Check response format compliance."""
    scores = []
    for completion in completions:
        response = completion[0]['content']
        format_check = check_xml_format(response)
        
        # Score based on format compliance
        score = 0.0
        if format_check['has_reasoning']:
            score += 0.2
        if format_check['has_answer']:
            score += 0.2
        if format_check['has_newlines']:
            score += 0.05
        if format_check['proper_nesting']:
            score += 0.05
            
        scores.append(score)
    return scores

def reasoning_quality_reward(completions, **kwargs) -> List[float]:
    """Evaluate quality of reasoning steps."""
    scores = []
    for completion in completions:
        response = completion[0]['content']
        reasoning = extract_xml_reasoning(response)
        
        # Count reasoning steps
        steps = count_reasoning_steps(response)
        
        # Score based on reasoning quality
        score = 0.0
        if steps > 0:
            # Base score for having steps
            score += 0.2
            
            # Bonus for number of steps (up to 0.3)
            score += min(steps * 0.1, 0.3)
            
            # Check for mathematical notation if relevant
            if re.search(r'[=+\-*/\^]', reasoning):
                score += 0.2
            
            # Check for variable usage if relevant
            if re.search(r'[a-zA-Z]_?\{?[a-zA-Z0-9]*\}?', reasoning):
                score += 0.2
            
            # Cap at 1.0
            score = min(score, 1.0)
        
        scores.append(score)
    return scores

def step_clarity_reward(completions, **kwargs) -> List[float]:
    """Evaluate clarity and structure of solution steps."""
    scores = []
    for completion in completions:
        response = completion[0]['content']
        reasoning = extract_xml_reasoning(response)
        steps = reasoning.split('\n')
        
        score = 0.0
        if steps:
            # Check for numbered/ordered steps
            if any(re.match(r'^[0-9]+[).] ', step.strip()) for step in steps):
                score += 0.2
            
            # Check for clear step progression
            if any(word in reasoning.lower() for word in ['first', 'then', 'next', 'finally']):
                score += 0.2
            
            # Check for mathematical clarity
            if re.search(r'=.*=', reasoning):  # Multiple equality signs indicating step-by-step calculation
                score += 0.2
            
            # Check for explanatory text
            if re.search(r'[A-Za-z]{3,}', reasoning):  # At least some words in explanations
                score += 0.2
            
            # Penalize very long steps
            if max(len(step) for step in steps) > 200:
                score -= 0.1
        
        scores.append(max(0.0, score))  # Ensure non-negative score
    return scores

# Reward function weights
REWARD_WEIGHTS = {
    'correctness_reward': 1.0,
    'format_reward': 0.5,
    'reasoning_quality_reward': 0.3,
    'step_clarity_reward': 0.2
}
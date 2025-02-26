import re
from typing import Any, Dict, List, Callable, Optional
from pathlib import Path
import json
import ast
from datasets import load_dataset

class RewardFunctionGenerator:
    """Generates reward functions by analyzing dataset patterns."""
    
    def __init__(self, dataset_name: str, config: str = None):
        self.dataset_name = dataset_name
        self.config = config
        self.dataset = load_dataset(dataset_name, config)
        self.analysis = self._analyze_dataset()
        
    def _analyze_dataset(self) -> Dict[str, Any]:
        """Analyze dataset patterns to inform reward function generation."""
        train_data = self.dataset['train']
        sample_size = min(100, len(train_data))
        samples = train_data[:sample_size]
        
        analysis = {
            "has_numerical_answers": False,
            "has_text_answers": False,
            "has_multi_step": False,
            "has_variables": False,
            "has_equations": False,
            "answer_patterns": set(),
            "reasoning_patterns": set(),
            "typical_step_count": 0,
            "max_answer_length": 0,
            "columns": list(train_data[0].keys())
        }
        
        # Detect question/answer columns
        for col in analysis["columns"]:
            col_lower = col.lower()
            if any(q in col_lower for q in ['question', 'problem', 'input']):
                analysis["question_column"] = col
            elif any(a in col_lower for a in ['answer', 'solution', 'output']):
                analysis["answer_column"] = col
        
        # Analyze answer patterns
        answers = []
        for sample in samples:
            try:
                # Handle different dataset formats
                if isinstance(sample, dict):
                    if 'answers' in sample and isinstance(sample['answers'], dict):
                        # SQuAD format
                        answer = sample['answers']['text'][0]
                    else:
                        # Try common answer column names
                        for key in ['answer', 'solution', 'output']:
                            if key in sample:
                                answer = sample[key]
                                analysis["answer_column"] = key
                                break
                        else:
                            continue
                else:
                    continue
                
                # Convert answer to string and analyze
                answer_str = str(answer)
                answers.append(answer_str)
                
                # Check for numerical answers
                if re.match(r'^\s*-?\d+\.?\d*\s*$', answer_str):
                    analysis["has_numerical_answers"] = True
                else:
                    analysis["has_text_answers"] = True
                
                # Check for equations
                if re.search(r'[=+\-*/\^]', answer_str):
                    analysis["has_equations"] = True
                
                # Check for variables
                if re.search(r'[a-zA-Z]_?\{?[a-zA-Z0-9]*\}?', answer_str):
                    analysis["has_variables"] = True
                
                # Track max answer length
                analysis["max_answer_length"] = max(analysis["max_answer_length"], len(answer_str))
                
            except (KeyError, IndexError, AttributeError):
                continue
            
            # Check for numerical answers
            if re.match(r'^\s*-?\d+\.?\d*\s*$', str(answer)):
                analysis["has_numerical_answers"] = True
            else:
                analysis["has_text_answers"] = True
            
            # Check for equations
            if re.search(r'[=+\-*/\^]', str(answer)):
                analysis["has_equations"] = True
            
            # Check for variables
            if re.search(r'[a-zA-Z]_?\{?[a-zA-Z0-9]*\}?', str(answer)):
                analysis["has_variables"] = True
            
            # Track max answer length
            analysis["max_answer_length"] = max(analysis["max_answer_length"], len(str(answer)))
        
        return analysis
    
    def generate_reward_functions(self) -> Dict[str, Callable]:
        """Generate reward functions based on dataset analysis."""
        reward_functions = {}
        
        # Helper functions that will be used by reward functions
        helpers = '''
def extract_xml_answer(text: str) -> str:
    """Extract answer from XML tags."""
    match = re.search(r"<answer>[ \t]*(.*?)[ \t]*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_xml_reasoning(text: str) -> str:
    """Extract reasoning from XML tags."""
    match = re.search(r"<reasoning>[ \t]*(.*?)[ \t]*</reasoning>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def count_reasoning_steps(text: str) -> int:
    """Count the number of reasoning steps."""
    reasoning = extract_xml_reasoning(text)
    steps = [s.strip() for s in reasoning.split("\\n") if s.strip()]
    return len(steps)

def check_xml_format(text: str) -> Dict[str, bool]:
    """Check XML format compliance."""
    return {
        "has_reasoning": bool(re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL)),
        "has_answer": bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL)),
        "has_newlines": text.count("\\n") > 0,
        "proper_nesting": text.find("<reasoning>") < text.find("</reasoning>") < text.find("<answer>") < text.find("</answer>")
    }
'''
        
        # Basic correctness reward
        correctness_func = f"""
def correctness_reward(prompts, completions, sample_data, **kwargs) -> List[float]:
    \"\"\"Compare completion answer with ground truth.\"\"\"
    scores = []
    for completion in completions:
        response = completion[0]['content']
        extracted = extract_xml_answer(response)
        
        # Get ground truth answer
        answer_data = sample_data['{self.analysis["answer_column"]}']
        if isinstance(answer_data, dict) and 'text' in answer_data:
            truth = str(answer_data['text'][0]) if isinstance(answer_data['text'], list) else str(answer_data['text'])
        else:
            truth = str(answer_data)
            
        # Compare answers
        {'if self.analysis["has_numerical_answers"]:' if self.analysis["has_numerical_answers"] else ''}
        try:
            extracted_num = float(extracted)
            truth_num = float(truth)
            scores.append(2.0 if abs(extracted_num - truth_num) < 1e-6 else 0.0)
        except ValueError:
            scores.append(2.0 if extracted.strip() == truth.strip() else 0.0)
        {'else:' if self.analysis["has_numerical_answers"] else ''}
            scores.append(2.0 if extracted.strip() == truth.strip() else 0.0)
    return scores
"""
        reward_functions["correctness_reward"] = correctness_func

        # Format reward
        format_func = """
def format_reward(completions, **kwargs) -> List[float]:
    \"\"\"Check response format compliance.\"\"\"
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
"""
        reward_functions["format_reward"] = format_func

        # Reasoning quality reward
        reasoning_func = """
def reasoning_quality_reward(completions, **kwargs) -> List[float]:
    \"\"\"Evaluate quality of reasoning steps.\"\"\"
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
"""
        reward_functions["reasoning_quality_reward"] = reasoning_func

        # Step clarity reward
        step_clarity_func = """
def step_clarity_reward(completions, **kwargs) -> List[float]:
    \"\"\"Evaluate clarity and structure of solution steps.\"\"\"
    scores = []
    for completion in completions:
        response = completion[0]['content']
        reasoning = extract_xml_reasoning(response)
        steps = reasoning.split('\\n')
        
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
"""
        reward_functions["step_clarity_reward"] = step_clarity_func

        # Save reward functions to a Python file
        output_dir = Path("generated_rewards")
        output_dir.mkdir(exist_ok=True)
        
        safe_name = self.dataset_name.replace('/', '_').replace('-', '_')
        output_path = output_dir / f"{safe_name}_rewards.py"
        
        with open(output_path, 'w') as f:
            # Format the code properly
            code_parts = []
            
            # Imports
            code_parts.extend([
                "import re",
                "from typing import List, Dict",
                "",
                "# Helper functions",
                helpers.strip(),
                "",
                "# Reward functions",
                correctness_func.strip().replace("'", '"'),
                "",
                format_func.strip().replace("'", '"'),
                "",
                reasoning_func.strip().replace("'", '"').replace("[=+\\-*/\\^]", "[=+\\\\-*/\\\\^]"),
                "",
                step_clarity_func.strip().replace("'", '"'),
                "",
                "# Reward function weights",
                "REWARD_WEIGHTS = {",
                '    "correctness_reward": 1.0,',
                '    "format_reward": 0.5,',
                '    "reasoning_quality_reward": 0.3,',
                '    "step_clarity_reward": 0.2',
                "}"
            ])
            
            # Join code parts with proper line endings
            final_code = "\n".join(code_parts)
            
            # Fix function parameters and syntax
            replacements = {
                "(prompts completions": "(prompts, completions",
                "(completions **kwargs": "(completions, **kwargs",
                "Dict[str bool]": "Dict[str, bool]",
                "min(steps * 0.1 0.3": "min(steps * 0.1, 0.3)",
                "min(score 1.0": "min(score, 1.0)",
                "max(0.0 score": "max(0.0, score)",
                "re.search(r\"<answer>.*?</answer>\" text": "re.search(r\"<answer>.*?</answer>\", text",
                "re.search(r\"<reasoning>.*?</reasoning>\" text": "re.search(r\"<reasoning>.*?</reasoning>\", text",
                "re.search(r\"=.*=\" reasoning": "re.search(r\"=.*=\", reasoning",
                "re.search(r\"[A-Za-z]{3}\" reasoning": "re.search(r\"[A-Za-z]{3,}\", reasoning",
                "isinstance(answer_data dict": "isinstance(answer_data, dict",
                "isinstance(answer_data['text'] list": "isinstance(answer_data['text'], list",
                "re.match(r'^[0-9]+[).] ' step.strip()": "re.match(r'^[0-9]+[).] ', step.strip())",
                "['first' 'then' 'next' 'finally']": "['first', 'then', 'next', 'finally']",
                "r\"<answer>[ \\t]*(.*?)[ \\t]*</answer>\"": "r\"<answer>\\s*(.*?)\\s*</answer>\"",
                "r\"<reasoning>[ \\t]*(.*?)[ \\t]*</reasoning>\"": "r\"<reasoning>\\s*(.*?)\\s*</reasoning>\"",
                ", re.DOTALL text": ", text, re.DOTALL",
                "re.DOTALL text": "text, re.DOTALL"
            }
            
            final_code = code_parts
            for old, new in replacements.items():
                final_code = [part.replace(old, new) for part in final_code]
            final_code = "\n".join(final_code)
            
            # Write the formatted code
            f.write(final_code)
        
        return str(output_path)

    def save_analysis(self, output_dir: str = "reward_analysis") -> str:
        """Save dataset analysis results."""
        Path(output_dir).mkdir(exist_ok=True)
        safe_name = self.dataset_name.replace('/', '_').replace('-', '_')
        output_path = Path(output_dir) / f"{safe_name}_analysis.json"
        
        # Convert set to list for JSON serialization
        analysis_copy = self.analysis.copy()
        analysis_copy['answer_patterns'] = list(self.analysis['answer_patterns'])
        analysis_copy['reasoning_patterns'] = list(self.analysis['reasoning_patterns'])
        
        with open(output_path, 'w') as f:
            json.dump(analysis_copy, f, indent=2)
        
        return str(output_path)

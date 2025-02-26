# Open-R1-Math-220k Fine-tuning Implementation

This repository contains the implementation for fine-tuning a small language model (based on Qwen 2.5 .5B distilled) on the Open-R1-Math-220k dataset for mathematical reasoning.

## Project Structure

- `openr1_finetuning.py`: Main implementation file containing all the functions for dataset loading, preprocessing, model training, and evaluation.
- `test_openr1_dataset.py`: Script to test dataset loading and preprocessing functions.
- `test_model_setup.py`: Script to test model loading and training setup.
- `train_openr1_small.py`: Script to run a small-scale training test on a subset of the dataset.
- `requirements.txt`: List of required dependencies.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Test Dataset Loading and Preprocessing

To test the dataset loading and preprocessing functions:

```bash
python test_openr1_dataset.py
```

This script will:
- Load the Open-R1-Math-220k dataset
- Process a small subset of the dataset
- Create training and validation splits
- Test the answer extraction function

### 2. Test Model Setup

To test the model loading and training setup:

```bash
python test_model_setup.py
```

This script will:
- Load the model and tokenizer
- Prepare the model for training with LoRA
- Set up training arguments
- Test prompt formatting

### 3. Run Small-Scale Training

To run a small-scale training test on a subset of the dataset:

```bash
python train_openr1_small.py
```

This script will:
- Load a small subset of the dataset
- Preprocess the data
- Load and prepare the model
- Run a small training test (10 steps)
- Save the model
- Evaluate on a few examples

### 4. Full Training

To run the full training pipeline:

```bash
python openr1_finetuning.py
```

This script will:
- Load the full dataset
- Preprocess the data
- Load and prepare the model
- Run the full training
- Save the model
- Evaluate on the validation set

## Implementation Details

### Dataset Processing

The implementation extracts questions, answers, and reasoning steps from the Open-R1-Math-220k dataset. The dataset is expected to have the following structure:

- `instruction`: The question/problem
- `output`: The solution, with the answer at the end after the last "="

### Model Training

The implementation uses LoRA (Low-Rank Adaptation) for efficient fine-tuning of the model. This allows for training with less memory and fewer parameters.

### Evaluation

The model is evaluated based on the correctness of the extracted answers. The answer extraction function handles different formats:

1. Answers after the last "=" sign
2. Answers within XML tags (`<answer>...</answer>`)
3. The last line of the generated text (fallback)

## Customization

You can customize the training by modifying the following parameters in `openr1_finetuning.py`:

- `model_name`: The base model to use (default: "Qwen/Qwen2.5-0.5B")
- `num_train_epochs`: Number of training epochs (default: 1)
- `learning_rate`: Learning rate (default: 1e-6)
- `per_device_train_batch_size`: Batch size (default: 1)
- `gradient_accumulation_steps`: Gradient accumulation steps (default: 4)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Open-R1-Math-220k Dataset](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [Qwen 2.5 Models](https://huggingface.co/Qwen)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT Library](https://github.com/huggingface/peft) 
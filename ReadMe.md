# SmolAgent Fine-Tuning with the Open-R1-Math-220k Dataset (2% Training)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarleyCoops/smolThinker-.5B/blob/main/Qwen2.5-.5BOpenR1.ipynb)



## Project Update - Feb 13, 2025 

50% training is underway but I am noticing context window limits during training. In order to fit under 40GB of RAM, you need to train Qwen with a maximum token limit of roughly 256. Upon inference, you can ask for longer outputs, but the prompts are so long in these models that training requires a larger GPU than what is available on Colab.

This is a 50% run proof of concept using smaller context windows, a full R1 run with larger context windows will require an H100 which I will launch later today. 

For more details, see the run on [Weights & Biases](https://wandb.ai/christian-cooper-us/qwen-OpenR1math-50.00?nw=nwuserchristiancooperus).

## Overview

So what is the smallest language model that can be trained on how specific of a math problem? This is the first deployed version of the Open-R1-Math-220k dataset fine-tuned on the Qwen 2.5 0.5B distilled version.

This project demonstrates a 2% trained version of a small language model (0.5B parameters) on the Open-R1-Math-220k dataset. Building on the Qwen 2.5 0.5B distilled version, I've fine-tuned the model on a subset of the data to explore how effectively a small model can learn mathematical reasoning. This is proof of concept for efficient fine-tuning with limited computational resources. I will train this over the next two days and see how far I get. 

The setup is simple. You need a Colab Pro account with A100 GPU access. You need 40 GB to train this model. Run each cell, and eventually publish the fully trained model to your HuggingFace account. The python script can be used to call back to your model for inference, through the terminal. 

You can then deploy with HF inference to set up api calls to your endpoint. 

## Project Goals

- **Enhanced Mathematical Reasoning:** Enable the model to perform complex mathematical reasoning using chain-of-thought processes with the Open-R1-Math-220k dataset.
- **Efficient Small Model Training:** Explore the capabilities of a 0.5B parameter model on focused mathematical tasks.
- **Scalable Training Process:** Document the training process from 2% to full dataset training.
- **SmolAgent Integration:** Integrate with the HuggingFace SmolAgent class for seamless deployment and inference.

## Workflow and Implementation Steps

### 1. Access and Load the Dataset

- **Dataset Information:**  
  The Open-R1-Math-220k dataset is available on Hugging Face as `open-r1-math-220k`.

- **Loading the Dataset:**  
  ```python
  from datasets import load_dataset

  openr1_dataset = load_dataset("open-r1-math-220k")
  print(openr1_dataset)
  print(openr1_dataset['train'][0])
  ```

### 2. Dataset Structure

- **Key Fields:**  
  Each example in the Open-R1-Math-220k dataset contains:
  - **Problem Statement:** The mathematical question or problem
  - **Chain-of-Thought Reasoning:** Step-by-step solution process
  - **Final Answer:** The numerical or textual solution

### 3. Training Process

- **Initial Training (2%):**
  - Using a subset of the Open-R1-Math-220k dataset
  - Training on A100 GPU with 40GB memory
  - Monitoring progress with Weights & Biases

- **Scaling Up:**
  - Progressive training on larger portions of the dataset
  - Checkpoint saving every 100 steps
  - Performance monitoring and validation

### 4. Deployment

- **Model Publishing:**
  - Upload trained model to HuggingFace Hub
  - Set up inference endpoint
  - Terminal-based inference using provided Python script

### 5. Inference

- **Local Usage:**
  - Run inference through terminal commands
  - Get step-by-step reasoning and solutions
  - Save results in JSONL format

- **API Integration:**
  - HuggingFace Inference API setup
  - Endpoint configuration
  - Response formatting

## Getting Started

1. **Setup Requirements:**
   - Colab Pro account
   - A100 GPU access (40GB memory)
   - HuggingFace account for model hosting

2. **Training:**
   - Open the Colab notebook
   - Run setup cells
   - Monitor training progress
   - Save checkpoints

3. **Deployment:**
   - Push model to HuggingFace
   - Configure inference endpoint
   - Test with provided Python script

## Contributing

Contributions to improve the training process, inference pipeline, and documentation are welcome. Feel free to fork the repository and submit pull requests.

## License

This project is open-sourced under the MIT License.

## Acknowledgements

- Thanks to the Hugging Face team for the infrastructure and tools
- Appreciation to the creators of the Open-R1-Math-220k dataset
- Thanks to the Qwen team for the base model

---

This README outlines the project's purpose and the detailed steps required to train and deploy a small but effective math reasoning model. As the training progresses beyond 2%, updates will be made to reflect new findings and improvements.
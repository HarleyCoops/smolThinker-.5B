# SmolAgent Fine-Tuning with the Open-R1-Math-220k Dataset (2% Training)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Qwen2.5-.5BOpenR1.ipynb)

## Overview

This project demonstrates a 2% trained version of a small language model (0.5B parameters) on the Open-R1-Math-220k dataset. Building on the Qwen 2.5 0.5B distilled version, we've fine-tuned the model on a subset of the data to explore how effectively a small model can learn mathematical reasoning. This serves as a proof of concept for efficient fine-tuning with limited computational resources.

## Project Goals

- **Enhanced Reasoning:** Enable the model to perform complex mathematical reasoning using chain-of-thought processes.
- **Data Leverage:** Utilize the larger Open-R1-Math-220k dataset (220k examples) to further fine-tune and improve model performance.
- **Efficient Fine-Tuning:** Continue fine-tuning from an already distilled model to retain learned knowledge while pushing reasoning capabilities further.
- **SmolAgent Integration:** Integrate with the HuggingFace SmolAgent class, ensuring the refined model seamlessly interacts with smolagent pipelines.

## Workflow and Implementation Steps

### 1. Access and Load the Dataset

- **Dataset Information:**  
  Confirm the full identifier on Hugging Face (assumed here as `open-r1-math-220k`).

- **Loading the Dataset:**  
  Utilize the Hugging Face `datasets` library to load the dataset and inspect its structure.

  ```python
  from datasets import load_dataset

  openr1_dataset = load_dataset("open-r1-math-220k")
  print(openr1_dataset)
  print(openr1_dataset['train'][0])
  ```

### 2. Understand the Dataset Structure

- **Key Fields Identification:**  
  Assess each example to locate:
  - **Problem Statement:** Typically stored under keys like `"instruction"`, `"input"`, or similar.
  - **Chain-of-Thought Reasoning:** Potentially found under keys like `"rationale"`, `"reasoning"`, or `"chain_of_thought"`.
  - **Final Answer:** Expected keys include `"answer"`, `"solution"`, or `"output"`.

- **XML Format Consistency:**  
  Since the model was originally trained to generate outputs in an XML format (with `<reasoning>` and `<final_answer>` tags), ensure that the dataset either already conforms to or can be easily converted into this structure.

### 3. Adapt Data Preprocessing

- **Create a New Data Loader:**  
  Develop a function, for instance, `get_openr1_questions`, modeled after the existing `get_gsm8k_questions`, to extract the necessary fields from each example.

  ```python
  def get_openr1_questions(dataset, split="train"):
      questions = []
      answers = []
      reasonings = []

      for example in dataset[split]:
          # Adjust the field names based on your dataset's specifics.
          question = example["instruction"]  
          output = example["output"]

          # Locate the answer boundary using the last occurrence of '='
          answer_start_index = output.rfind("=") + 1
          reasoning = output[:answer_start_index]
          answer = output[answer_start_index:].strip()

          questions.append(question)
          answers.append(answer)
          reasonings.append(reasoning)

      return questions, answers, reasonings

  # Usage example:
  openr1_questions, openr1_answers, openr1_reasonings = get_openr1_questions(openr1_dataset)
  ```

### 4. Modify Training Data Loading

- **Update the Trainer:**  
  Replace references to the GSM8K dataset with data extracted from the Open-R1 dataset. When setting up the `GRPOTrainer`, pass the processed questions (and corresponding evaluation data) accordingly.

  ```python
  train_questions, train_answers, train_reasonings = get_openr1_questions(openr1_dataset)

  trainer = GRPOTrainer(
      model,
      args=training_args,
      train_dataset=train_questions,  # Directly use the questions
      eval_dataset=val_questions,
      tokenizer=tokenizer,
      compute_metrics=None,  # Metrics handled via custom reward functions
      grpo_loss_kwargs={
          "answer_grounding_loss_weight": answer_grounding_loss_weight,
      }
  )
  ```

### 5. Reward Functions and Answer Extraction

- **Primary Reward:**  
  The `correctness_reward_func` compares the model's extracted answer with the ground truth using a small tolerance to account for floating-point inaccuracies.

- **Adapting Answer Extraction:**  
  Modify or create an extraction function (e.g., `extract_openr1_answer`) to adapt to the output format from Open-R1.

  ```python
  def extract_openr1_answer(generated_text):
      answer_start_index = generated_text.rfind("=") + 1
      return generated_text[answer_start_index:].strip()
  ```

- **Format-Related Rewards:**  
  If the Open-R1 dataset maintains a consistent reasoning structure, you can simplify or even reduce reliance on strict format reward functions.

### 6. Adjust Training Hyperparameters

- **Epochs & Learning Rate:**  
  Given the larger dataset, consider starting with one epoch to monitor overfitting, and experiment with a reduced learning rate (e.g., 1e-6).
- **Batch Size & Accumulation:**  
  Maintain a `per_device_train_batch_size` of 1 and adjust `gradient_accumulation_steps` based on your GPU capacity.
- **Exploration Steps:**  
  Keep `num_generations` at 16 to ensure ample exploration.

- **Validation:**  
  Use the dataset's validation split for continuous evaluation and early stopping to prevent overfitting.

### 7. Training and Evaluation

- **Execute Training:**  
  Run the modified training script with the Open-R1 data. Monitor both the loss and reward metrics through tools like Weights & Biases.
- **Iterative Testing:**  
  Begin with a small subset (e.g., 100 examples) to ensure proper data extraction and processing, and then scale up.

### 8. Inference Pipeline

- **Seamless Integration:**  
  The inference pipeline (e.g., in `hf_grpotuned_pipeline.py`) should remain largely unchanged, as long as the extraction function correctly parses model outputs according to the expected XML format.

## Debugging and Iteration

- **Verbose Logging:**  
  Print processed data, extracted answers, and reward values to verify that each step is functioning as expected.
- **Incremental Development:**  
  Start with a limited number of examples to quickly iterate over changes. After verifying functionality on a small scale, move on to the complete dataset.
- **Hyperparameter Tuning:**  
  Experiment with different training configurations and reward weights to observe impacts on model performance.

## Contributing

Contributions to improve the data processing functions, training pipeline, and reward mechanisms are welcome. Fork the repository, make your modifications, and open an issue or pull request for discussion and inclusion.

## License

This project is open-sourced under the MIT License.

## Acknowledgements

- Thanks to the Hugging Face team for the `datasets` library.
- Appreciation to the contributors of the Open-R1-Math-220k dataset.
- Gratitude to community members exploring advanced chain-of-thought reasoning methodologies.

---

This README outlines the project's purpose and the detailed steps required to integrate and fine-tune a smaller model using the Open-R1-Math-220k dataset. As the project evolves, feel free to adapt these instructions and contribute improvements.
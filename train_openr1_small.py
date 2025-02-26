"""
Small-scale training script for Open-R1-Math-220k dataset.

This script runs a small-scale training test on a subset of the Open-R1-Math-220k dataset.
"""

import os
import torch
from datasets import load_dataset
from transformers import Trainer
from openr1_finetuning import (
    load_and_explore_dataset,
    preprocess_dataset,
    create_train_val_split,
    load_model_and_tokenizer,
    prepare_model_for_training,
    setup_training_arguments,
    prepare_training_data,
    evaluate_model
)

def main():
    """Main function to run a small-scale training test."""
    print("=" * 50)
    print("Small-scale Training Test on Open-R1-Math-220k Dataset")
    print("=" * 50)
    
    # Step 1: Load dataset
    print("\n1. Loading dataset...")
    dataset = load_and_explore_dataset()
    
    # Step 2: Create a small subset for testing
    print("\n2. Creating a small subset for testing...")
    num_train_examples = 100  # Use a small number for testing
    num_val_examples = 20
    
    small_dataset = {
        'train': dataset['train'].select(range(min(num_train_examples, len(dataset['train']))))
    }
    if 'validation' in dataset:
        small_dataset['validation'] = dataset['validation'].select(range(min(num_val_examples, len(dataset['validation']))))
    
    # Step 3: Preprocess the small subset
    print("\n3. Preprocessing the small subset...")
    train_questions, train_answers, train_reasonings, val_questions, val_answers, val_reasonings = create_train_val_split(small_dataset)
    
    # Step 4: Load model and tokenizer
    print("\n4. Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Step 5: Prepare model for training
    print("\n5. Preparing model for training...")
    model = prepare_model_for_training(model)
    
    # Step 6: Set up training arguments for a very small test run
    print("\n6. Setting up training arguments...")
    training_args = setup_training_arguments(output_dir="./results_small_test")
    
    # Override some parameters for a quick test
    training_args.num_train_epochs = 1
    training_args.max_steps = 10  # Just run for 10 steps
    training_args.logging_steps = 1
    training_args.eval_steps = 5
    training_args.save_steps = 10
    training_args.evaluation_strategy = "steps"
    
    # Step 7: Prepare training data
    print("\n7. Preparing training data...")
    train_data = prepare_training_data(train_questions, train_answers, train_reasonings, tokenizer)
    val_data = prepare_training_data(val_questions, val_answers, val_reasonings, tokenizer)
    
    # Create a custom dataset class
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    
    # Step 8: Set up trainer
    print("\n8. Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Step 9: Run a small training test
    print("\n9. Running a small training test...")
    try:
        trainer.train()
        print("\nTraining completed successfully!")
        
        # Step 10: Save the model
        print("\n10. Saving the model...")
        trainer.save_model("./openr1_small_test_model")
        
        # Step 11: Evaluate on a few examples
        print("\n11. Evaluating on a few examples...")
        evaluate_model(model, tokenizer, val_questions, val_answers, num_examples=5)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
    
    print("\nSmall-scale training test completed!")

if __name__ == "__main__":
    main() 
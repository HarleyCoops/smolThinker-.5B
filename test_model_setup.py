"""
Test script for model loading and training setup.

This script tests the model loading and training setup functions from openr1_finetuning.py.
"""

import torch
from openr1_finetuning import (
    load_model_and_tokenizer,
    prepare_model_for_training,
    setup_training_arguments,
    format_prompt
)

def main():
    """Main function to test model loading and training setup."""
    print("=" * 50)
    print("Testing Model Loading and Training Setup")
    print("=" * 50)
    
    # Check if CUDA is available
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Step 1: Test model and tokenizer loading
    print("\n1. Testing model and tokenizer loading...")
    try:
        model, tokenizer = load_model_and_tokenizer()
        print(f"Model loaded successfully: {model.__class__.__name__}")
        print(f"Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
        
        # Print model info
        print(f"\nModel config:")
        print(f"- Model type: {model.config.model_type}")
        print(f"- Vocab size: {model.config.vocab_size}")
        print(f"- Hidden size: {model.config.hidden_size}")
        print(f"- Num layers: {model.config.num_hidden_layers}")
        print(f"- Num attention heads: {model.config.num_attention_heads}")
        
        # Test tokenizer
        print("\nTesting tokenizer:")
        test_text = "Solve the following math problem: 2 + 2 = ?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"Input: '{test_text}'")
        print(f"Tokenized length: {tokens.input_ids.shape[1]}")
        print(f"Decoded: '{tokenizer.decode(tokens.input_ids[0])}'")
        
        # Step 2: Test model preparation for training
        print("\n2. Testing model preparation for training...")
        try:
            peft_model = prepare_model_for_training(model)
            print(f"Model prepared for training successfully: {peft_model.__class__.__name__}")
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in peft_model.parameters())
            print(f"\nTrainable parameters: {trainable_params:,} ({trainable_params/all_params:.2%} of total)")
            
        except Exception as e:
            print(f"Error preparing model for training: {str(e)}")
        
        # Step 3: Test training arguments setup
        print("\n3. Testing training arguments setup...")
        training_args = setup_training_arguments()
        print(f"Training arguments set up successfully")
        print(f"- Output directory: {training_args.output_dir}")
        print(f"- Learning rate: {training_args.learning_rate}")
        print(f"- Batch size: {training_args.per_device_train_batch_size}")
        print(f"- Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"- Number of epochs: {training_args.num_train_epochs}")
        
        # Step 4: Test prompt formatting
        print("\n4. Testing prompt formatting...")
        test_question = "If a train travels at 60 km/h, how far will it travel in 2.5 hours?"
        formatted_prompt = format_prompt(test_question)
        print(f"Original question: {test_question}")
        print(f"Formatted prompt:\n{formatted_prompt}")
        
        # Test tokenizing the prompt
        prompt_tokens = tokenizer(formatted_prompt, return_tensors="pt")
        print(f"Prompt token length: {prompt_tokens.input_ids.shape[1]}")
        
    except Exception as e:
        print(f"Error loading model and tokenizer: {str(e)}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 
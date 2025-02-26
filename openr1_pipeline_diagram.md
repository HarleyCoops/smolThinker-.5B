# OpenR1 Fine-tuning Pipeline Diagram

The following diagram illustrates the complete pipeline for fine-tuning the OpenR1 model on the Open-R1-Math-220k dataset, from data loading to visualization in Weights & Biases.

```mermaid
graph TD
    %% Main data flow
    A[Open-R1-Math-220k Dataset] --> B[Data Loading]
    B --> C[Data Preprocessing]
    C --> D[Training/Validation Split]
    
    D --> E1[Training Data]
    D --> E2[Validation Data]
    
    F[Base Model: Qwen 2.5 .5B] --> G[Model Loading]
    G --> H[Model Preparation with LoRA]
    
    E1 --> I[Format Training Data]
    E2 --> J[Format Validation Data]
    
    I --> K[Training Setup]
    H --> K
    
    K --> L[Training Process]
    L --> M[Trained Model]
    
    M --> N1[Answer Correctness Evaluation]
    M --> N2[Step-by-Step Reasoning Evaluation]
    
    E2 --> N1
    E2 --> N2
    
    %% Weights & Biases integration
    L -- "Training Metrics" --> O[Weights & Biases]
    N1 -- "Accuracy Metrics" --> O
    N2 -- "Reasoning Quality Metrics" --> O
    
    %% Reward Functions
    P1[Correctness Reward Function] --> N1
    P2[Step-by-Step Reward Function] --> N2
    
    %% Detailed components
    subgraph "Data Processing"
        B
        C
        D
    end
    
    subgraph "Model Preparation"
        G
        H
    end
    
    subgraph "Training"
        K
        L
        M
    end
    
    subgraph "Evaluation"
        N1
        N2
        P1
        P2
    end
    
    subgraph "Visualization"
        O
    end
    
    %% Additional details
    Q1[Answer Extraction] --> P1
    Q2[Answer Normalization] --> P1
    Q3[Answer Comparison] --> P1
    
    R1[Step Count Analysis] --> P2
    R2[Equation Presence Check] --> P2
    
    %% Wandb specific components
    S1[Custom WandbCallback] --> O
    L --> S1
    
    %% Final model output
    M --> T[Saved Fine-tuned Model]
    
    %% Style
    classDef process fill:#f9f,stroke:#333,stroke-width:2px;
    classDef data fill:#bbf,stroke:#333,stroke-width:2px;
    classDef model fill:#bfb,stroke:#333,stroke-width:2px;
    classDef evaluation fill:#fbb,stroke:#333,stroke-width:2px;
    classDef visualization fill:#fbf,stroke:#333,stroke-width:2px;
    
    class A,E1,E2 data;
    class B,C,D,I,J,Q1,Q2,Q3,R1,R2 process;
    class F,G,H,K,L,M,T model;
    class N1,N2,P1,P2 evaluation;
    class O,S1 visualization;
```

## Pipeline Stages

1. **Data Processing**
   - Load the Open-R1-Math-220k dataset
   - Preprocess the data to extract problems, solutions, and answers
   - Create training and validation splits

2. **Model Preparation**
   - Load the base Qwen 2.5 .5B model and tokenizer
   - Prepare the model for training with LoRA (Low-Rank Adaptation)
   - Configure the model for efficient fine-tuning

3. **Training**
   - Format the training and validation data with prompts and solutions
   - Set up training arguments and the trainer
   - Train the model using the prepared data
   - Save the fine-tuned model

4. **Evaluation**
   - Evaluate answer correctness using extraction and comparison functions
   - Evaluate step-by-step reasoning quality
   - Calculate metrics for model performance

5. **Visualization**
   - Initialize Weights & Biases for experiment tracking
   - Log training metrics during the training process
   - Log final evaluation metrics
   - Visualize all metrics in the Weights & Biases dashboard

## Key Components

- **Reward Functions**: Functions that evaluate the model's outputs
  - Correctness reward: Evaluates if the extracted answer matches the reference
  - Step-by-step reward: Evaluates the quality of the reasoning process

- **Answer Processing**:
  - Extraction: Extract answers from generated text
  - Normalization: Standardize answers for comparison
  - Comparison: Compare generated answers with reference answers

- **Weights & Biases Integration**:
  - Custom callback for logging metrics
  - Real-time visualization of training progress
  - Final performance metrics dashboard

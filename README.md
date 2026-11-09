# Comparative Analysis of Base Jamba and LoRA Fine-Tuned Jamba Models for Machine Learning and Deep Learning Question Answering

## 1. Project Overview

This project focuses on comparing a pre-trained Jamba language model with its LoRA fine-tuned version for answering questions related to Machine Learning (ML) and Deep Learning (DL).

The main objective is to investigate whether parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) can improve the model's ability to generate domain-specific answers.

The project also explores an LLMOps-oriented workflow covering data preparation, model fine-tuning, evaluation, experiment tracking and deployment.

## 2. Project Objectives

- Understand and work with the Jamba hybrid language model.
- Prepare a question-answering dataset related to Machine Learning and Deep Learning.
- Perform dataset validation and cleaning.
- Fine-tune the pre-trained model using LoRA.
- Evaluate the model using ROUGE and BERTScore metrics.
- Compare the base model and the LoRA fine-tuned model.
- Explore an LLMOps workflow for model management, serving.

## 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model Framework | Hugging Face Transformers |
| Fine-Tuning | PEFT, LoRA |
| Dataset Processing | Pandas |
| Evaluation | ROUGE, BERTScore |
| Development Environment | Google Colab, PyCharm |
| Experiment Tracking | MLflow (planned / as implemented) |
| API Development | FastAPI (planned / as implemented) |


## 4. Dataset Description

The project uses the Hugging Face dataset:

**Dataset:** [Koushim/qa-ml-dl-jsonl](https://huggingface.co/datasets/Koushim/qa-ml-dl-jsonl)

The dataset contains question-answer pairs related to Machine Learning and Deep Learning.

### Dataset Fields

| Field | Description |
|---|---|
| question | Contains the question |
| answer | Contains the corresponding answer |
| topic | Represents the subject category |
| level | Represents the difficulty level |

## 5. Data Preparation and Cleaning

The dataset preparation stage focuses on improving data quality before model fine-tuning.

The following checks and operations were included in the data preparation workflow:

- Loading the dataset using the Hugging Face Datasets library.
- Converting the dataset into Pandas DataFrames.
- Checking for missing and empty questions and answers.
- Identifying duplicate question-answer records.
- Checking questions associated with multiple answers.
- Normalizing text for duplicate detection.
- Splitting the dataset into training, validation and testing sets.
- Saving the processed datasets in Parquet format.

The intended data split is:

- Training: 80%
- Validation: 10%
- Testing: 10%

The final record counts should be updated according to the cleaned dataset.

## 6. Model Architecture

### Jamba

Jamba is a hybrid language model architecture that combines Transformer and Mamba components. It uses a combination of attention-based and state-space-model-based layers.

The project investigates Jamba as the underlying language model for Machine Learning and Deep Learning question answering.

### LoRA Fine-Tuning

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique.

Instead of updating all the parameters of the pre-trained model, LoRA introduces trainable low-rank adapter matrices into selected model layers.

This approach reduces the number of trainable parameters compared with full fine-tuning.

The project compares:

1. Base Jamba Model – the original pre-trained model.
2. LoRA Fine-Tuned Jamba Model – the model adapted using the project dataset.

## 7. Model Evaluation

The models are evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| ROUGE-1 | Measures unigram overlap |
| ROUGE-2 | Measures bigram overlap |
| ROUGE-L | Measures longest common subsequence |
| BERTScore F1 | Measures semantic similarity between generated and reference answers |

### Fine-Tuned Model Evaluation Results

The following are the evaluation results obtained for the LoRA fine-tuned model.

#### Training Evaluation

| Metric | Score |
|---|---:|
| ROUGE-1 | 0.11420 |
| ROUGE-2 | 0.04363 |
| ROUGE-L | 0.07796 |
| BERTScore F1 | 0.80952 |

#### Testing Evaluation

| Metric | Score |
|---|---:|
| ROUGE-1 | 0.11186 |
| ROUGE-2 | 0.04296 |
| ROUGE-L | 0.07479 |
| BERTScore F1 | 0.80627 |

### Base Model Comparison

The base model must be evaluated using the same evaluation dataset, generation settings and metric calculation procedure.

As we see that the fine-tuned model is perform better than not fine tuned model and also as we see that since BERTScore F1 compares the outputs
by changing them into embeddings so even words are changed but their respective meanings does not so only we get high score in BERTScore F1 but not in
other metrics

## 8. LLMOps Workflow

The project follows an LLMOps-oriented lifecycle:

1. Dataset preparation and validation.
2. Pre-trained Jamba model loading.
3. LoRA fine-tuning.
4. Model evaluation.
5. Experiment tracking using MLflow.
6. Model version management using a model registry.
7. API development using FastAPI.


The implementation status of each component should be updated as development progresses.

## 9. Project Workflow

```text
        QA Dataset
             |
             v
    Data Validation
             |
             v
     Data Cleaning
             |
             v
    Dataset Splitting
             |
             v
     Pre-trained Jamba
             |
       +-----+------+
       |            |
       v            v
   Base Model    LoRA Fine-Tuning
       |            |
       v            v
   Evaluation    Fine-Tuned Model
       |            |
       |            v
       |         Evaluation
       |            |
       +-----+------+
             |
             v
      Model Comparison
             |
             v
       LLMOps Workflow
             |
             v
    Serving and Monitoring
```

## 10. Expected Outcomes

The project aims to:

- Understand the working of a hybrid Jamba language model.
- Develop a domain-specific question-answering model.
- Investigate the effect of LoRA fine-tuning on model performance.
- Compare the base and fine-tuned models using quantitative evaluation metrics.
- Establish an LLMOps workflow for experiment management and model deployment.

## 11. Future Enhancements

Possible future enhancements include:

- Completing the base-model evaluation and comparative analysis.
- Improving answer quality through hyperparameter experimentation.
- Integrating Retrieval-Augmented Generation (RAG).
- Completing API-based model serving.
- Implementing experiment tracking and model versioning.
- Adding monitoring dashboards for model-serving performance.

## 12. Conclusion

This project explores the application of Jamba for Machine Learning and Deep Learning question answering and investigates the effect of LoRA-based parameter-efficient fine-tuning.

By comparing the pre-trained model with its fine-tuned version, the project aims to understand how domain-specific adaptation affects answer quality.

The LLMOps workflow further provides a foundation for managing, serving and monitoring language models in a practical environment.

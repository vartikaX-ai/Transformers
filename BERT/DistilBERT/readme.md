# DistilBERT Sentiment Analysis

## Overview

This project demonstrates sentiment classification using **DistilBERT**, a smaller and faster Transformer-based language model distilled from BERT.

A pre-trained `distilbert-base-uncased` model was fine-tuned on a review dataset to classify reviews into two sentiment classes:

- Negative
- Positive

## Model

- **Model:** DistilBERT
- **Pre-trained Model:** `distilbert-base-uncased`
- **Architecture:** Distilled Encoder-only Transformer
- **Task:** Binary Sentiment Classification
- **Number of Classes:** 2
- **Maximum Sequence Length:** 128 tokens
- **Learning Rate:** 2e-5

## Dataset

The dataset contains **50,000 reviews**:

- 25,000 Positive reviews
- 25,000 Negative reviews

For this practical, a subset of the dataset was used to reduce training time.

### Data Split

| Dataset | Samples |
|--------|--------:|
| Training | 4,500 |
| Validation | 500 |
| Testing | 1,000 |
| **Total Used** | **6,000** |

## Preprocessing

The text data was tokenized using the DistilBERT tokenizer.

The tokenizer generates:

- Input IDs
- Attention Masks

The sequences were:

- Truncated to a maximum length of 128 tokens
- Padded where required

The sentiment labels were encoded for binary classification.

## DistilBERT Fine-Tuning

The pre-trained DistilBERT model was fine-tuned by adding a classification head for sentiment classification.

The model was trained using:

- **Optimizer:** Adam
- **Learning Rate:** 2e-5
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Epochs:** 10

## Results

The model was evaluated on **1,000 unseen test reviews**.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------|----------:|-------:|---------:|--------:|
| 0 | 0.82 | 0.78 | 0.80 | 520 |
| 1 | 0.78 | 0.81 | 0.79 | 480 |
| **Overall Accuracy** | | | **0.796** | **1000** |

### Performance

- **Accuracy:** 79.6%
- **Macro F1-Score:** 0.80
- **Weighted F1-Score:** 0.80

The model achieved balanced performance across both sentiment classes, with an F1-score of **0.80** overall.

## Technologies Used

- Python
- TensorFlow
- Keras
- Hugging Face Transformers
- Scikit-learn
- NumPy
- Pandas

## Project Structure

```text
DistilBERT/
│
├── DistilBERT_Project/
│   └── DistilBERT.ipynb
│
└── README.md
```

## Key Concepts Demonstrated

- DistilBERT architecture
- Knowledge distillation
- DistilBERT tokenizer
- Transformer-based text representation
- Attention masks
- Transfer learning
- Fine-tuning a pre-trained language model
- Binary sentiment classification
- Model evaluation
- Precision, Recall and F1-score

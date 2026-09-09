# BERT Sentiment Analysis

## Overview

This project demonstrates sentiment classification using **BERT (Bidirectional Encoder Representations from Transformers)**.

A pre-trained `bert-base-uncased` model was fine-tuned on a review dataset to classify reviews into two sentiment classes:

- Negative
- Positive

## Model

- **Model:** BERT
- **Pre-trained Model:** `bert-base-uncased`
- **Architecture:** Encoder-only Transformer
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

The text data was tokenized using the BERT tokenizer.

The tokenizer generates:

- Input IDs
- Attention Masks

The sequences were:

- Truncated to a maximum length of 128 tokens
- Padded where required

The sentiment labels were encoded for binary classification.

## BERT Fine-Tuning

The pre-trained BERT model was fine-tuned by adding a classification head on top of BERT.

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
| Negative | 0.84 | 0.89 | 0.87 | 470 |
| Positive | 0.90 | 0.85 | 0.87 | 530 |
| **Overall Accuracy** | | | **0.87** | **1000** |

### Performance

- **Accuracy:** 87%
- **Macro F1-Score:** 0.87
- **Weighted F1-Score:** 0.87

The model achieved balanced performance across both sentiment classes, with an F1-score of **0.87** for both Negative and Positive reviews.

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
BERT/
│
├── BERT_Project/
│   └── bert_sentiment_analysis.ipynb
│
└── README.md
```

## Key Concepts Demonstrated

- BERT architecture
- BERT tokenizer
- Transformer-based text representation
- Attention masks
- Transfer learning
- Fine-tuning a pre-trained language model
- Binary sentiment classification
- Model evaluation
- Precision, Recall and F1-score

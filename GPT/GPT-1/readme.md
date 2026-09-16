# GPT-1 Text Generation using Hugging Face and TensorFlow

## 📌 Project Overview

This project demonstrates **text generation using GPT-1 (Generative Pre-trained Transformer 1)** with Hugging Face Transformers and TensorFlow.

GPT-1 is a decoder-only Transformer-based language model that generates text using autoregressive next-token prediction. The model predicts the next token based on the previously provided context and generated tokens.

In this practical, different decoding strategies are implemented and compared to understand how they influence text generation.

---

## 🎯 Objectives

* Load a pretrained GPT-1 model using Hugging Face Transformers.
* Tokenize input text using the GPT-1 tokenizer.
* Understand input IDs and attention masks.
* Generate text using different decoding strategies.
* Compare deterministic and probabilistic text generation.
* Understand the effect of temperature, top-k, and top-p sampling.

---

## 🧠 Model Used

**Model:** `openai-community/openai-gpt`

**Frameworks and Libraries:**

* Python
* TensorFlow
* Hugging Face Transformers
* NumPy

GPT-1 is an autoregressive, decoder-only language model that generates text by predicting one token at a time.

---

## 🔄 Text Generation Workflow

```text
Input Prompt
     ↓
Tokenization
     ↓
Input IDs and Attention Mask
     ↓
GPT-1 Model
     ↓
Next-Token Prediction
     ↓
Decoding Strategy
     ↓
Generated Token IDs
     ↓
Token Decoding
     ↓
Generated Text
```

---

## 🛠️ Technologies Used

| Technology                | Purpose                        |
| ------------------------- | ------------------------------ |
| Python                    | Programming language           |
| TensorFlow                | Deep learning framework        |
| Hugging Face Transformers | Pretrained model and tokenizer |
| NumPy                     | Numerical data handling        |
| GPT-1                     | Autoregressive text generation |

---

## 🚀 Installation

Install the required libraries:

```bash
pip install tensorflow transformers=4.50.0 numpy
```

### Import Libraries

```python
import numpy as np
import tensorflow as tf

from transformers import (
    TFAutoModelForCausalLM,
    AutoTokenizer
)
```
---

## 📊 Observations

Different decoding strategies produced different text continuations.

| Decoding Strategy    | Observation                                                       |
| -------------------- | ----------------------------------------------------------------- |
| Greedy Decoding      | Selected the highest-probability token at each step               |
| Temperature Sampling | Generated a different continuation through probabilistic sampling |
| Top-k Sampling       | Restricted the candidate pool to the top-k tokens                 |
| Top-p Sampling       | Restricted candidates using cumulative probability                |

The generated outputs included fictional dialogue and unrelated statements. This demonstrates that **text generation fluency does not guarantee factual accuracy or semantic correctness**.

The results depend on the model, prompt, decoding parameters, and sampling randomness.

---

## 📚 Key Concepts Learned

* Autoregressive language modeling
* Next-token prediction
* Tokenization
* Input IDs
* Attention masks
* Greedy decoding
* Temperature sampling
* Top-k sampling
* Top-p sampling
* Causal language models
* Hugging Face Transformers
* TensorFlow model inference

# BART — Text Summarization

## Overview

This practical explores **BART (Bidirectional and Auto-Regressive Transformers)** using Hugging Face Transformers.

BART is an **encoder-decoder Transformer** pretrained as a **denoising autoencoder**. The encoder processes corrupted input text, while the decoder reconstructs the original text.

In this practical, BART is used for **abstractive text summarization**.

---

## Model Used

```text
facebook/bart-large-cnn
```

`facebook/bart-large-cnn` is a BART model fine-tuned for summarization.

---

## Concepts Covered

* BART architecture
* Encoder-decoder Transformer
* Bidirectional encoder
* Autoregressive decoder
* Self-attention
* Cross-attention
* Denoising autoencoder pretraining
* Text corruption
* Abstractive summarization
* Beam search
* `num_beams`
* `min_length`
* `max_length`
* `early_stopping`
* Hugging Face Transformers

---

## BART Architecture

```text
Input Text
    ↓
Tokenizer
    ↓
BART Encoder
    ↓
Encoder Representations
    ↓
Cross-Attention
    ↓
BART Decoder
    ↓
Generated Summary
```

The encoder uses **bidirectional self-attention** to understand the input, while the decoder generates the summary autoregressively.

---

### 1. Import Libraries

```python
from transformers import BartTokenizer, TFBartForConditionalGeneration
```

### 2. Load Tokenizer and Model

```python
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

model = TFBartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn"
)
---

## Example

### Input

```text
Artificial intelligence has become an important part of modern technology.
Machine learning allows computers to learn patterns from data, while deep
learning uses neural networks to solve more complex problems. Generative AI
has further expanded these capabilities by allowing models to generate text,
images, audio, and code.
```

### Generated Summary

```text
Artificial intelligence includes machine learning and deep learning,
while generative AI enables models to create text, images, audio, and code.
```

The output is shorter than the original text while preserving its main ideas.

---

## Key Takeaways

* BART uses an **encoder-decoder Transformer architecture**.
* The encoder is **bidirectional**, while the decoder is **autoregressive**.
* BART is pretrained using a **denoising objective**.
* The decoder uses **cross-attention** to access encoder representations.
* `facebook/bart-large-cnn` is suitable for summarization.
* Beam search can improve sequence generation compared with greedy decoding.
* Hugging Face provides pretrained BART models that can be directly used for inference.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* Hugging Face Transformers

---

## Requirements

```text
transformers
tensorflow
```

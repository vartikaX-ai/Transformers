# GPT-2 Text Generation using Hugging Face and TensorFlow

## 📌 Overview

This project demonstrates text generation using **GPT-2**, a decoder-only Transformer-based language model, with Hugging Face Transformers and TensorFlow.

The practical explores autoregressive text generation and different decoding strategies, including greedy decoding, temperature sampling, top-k sampling, and top-p (nucleus) sampling.

It also demonstrates techniques for reducing repetitive text during generation.

---

## 🎯 Objectives

* Load a pretrained GPT-2 model using Hugging Face Transformers.
* Load and use the GPT-2 tokenizer.
* Understand input IDs and attention masks.
* Generate text using `Model.generate()`.
* Implement greedy decoding.
* Implement temperature sampling.
* Implement top-k sampling.
* Implement top-p sampling.
* Understand the effect of repetition penalties.
* Use `no_repeat_ngram_size` to reduce repeated phrases.
* Decode generated token IDs into human-readable text.

---

## 🧠 Concepts Covered

### 1. GPT-2

GPT-2 is a decoder-only Transformer language model that generates text autoregressively.

It predicts the next token based on the previously available tokens.

### 2. Tokenization

The tokenizer converts input text into numerical token IDs that can be processed by the model.

```python
inputs = Tokenizer(
    prompt,
    return_tensors="tf"
)
```

### 3. Input IDs

`input_ids` contain the numerical representation of the input tokens.

```python
inputs["input_ids"]
```

### 4. Attention Mask

The attention mask identifies valid input tokens and padding tokens.

* `1` → Real token
* `0` → Padding token

### 5. Autoregressive Generation

GPT-2 generates text one token at a time.

The newly generated token is added to the sequence and used to predict the next token.

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Hugging Face Transformers
* NumPy

---

## 📦 Model Used

```python
model_name = "gpt2"
```

The model and tokenizer are loaded using:

```python
from transformers import AutoTokenizer, TFAutoModelForCausalLM

Tokenizer = AutoTokenizer.from_pretrained(model_name)

Model = TFAutoModelForCausalLM.from_pretrained(model_name)
```

---

## 📊 Output Observations

The same prompt was used to compare different decoding strategies.

| Decoding Strategy    | Observation                                                     |
| -------------------- | --------------------------------------------------------------- |
| Greedy Decoding      | Generated a repetitive continuation                             |
| Temperature Sampling | Produced a different continuation with sampling-based variation |
| Top-k Sampling       | Generated text using a restricted candidate pool                |
| Top-p Sampling       | Generated text using a dynamic probability-based candidate pool |

**Note:** The quality and coherence of generated text depend on the model, prompt, and decoding parameters. Sampling does not guarantee factual accuracy or consistently better output.

---
## ✅ Key Learnings

* GPT-2 uses a decoder-only Transformer architecture.
* Text is converted into token IDs before being processed by the model.
* `Model.generate()` produces a sequence of token IDs.
* `Tokenizer.decode()` converts generated token IDs into readable text.
* Greedy decoding selects the highest-probability token.
* Temperature changes the probability distribution during sampling.
* Top-k restricts the candidate pool to the highest-ranked tokens.
* Top-p restricts candidates using cumulative probability.
* Repetition-control parameters can reduce repetitive text.

---


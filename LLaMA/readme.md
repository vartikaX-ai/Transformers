# LLaMA 3 Text Generation with Hugging Face

## Overview

This project demonstrates text generation using **Meta Llama 3 8B Instruct** with the Hugging Face Transformers library.

The practical focuses on understanding and implementing four commonly used decoding strategies:

* Greedy Decoding
* Temperature Sampling
* Top-k Sampling
* Top-p (Nucleus) Sampling

The same prompt is used with different decoding strategies to observe how generation behavior changes.

---

## Objective

The objective of this practical is to:

* Load Meta Llama 3 8B Instruct using Hugging Face Transformers
* Use 4-bit quantization to run the model within Google Colab GPU memory
* Perform text generation using different decoding strategies
* Compare deterministic and probabilistic generation
* Understand how temperature, top-k, and top-p control the diversity of generated text

---

## Model

**Model:** `meta-llama/Meta-Llama-3-8B-Instruct`

**Architecture:** Decoder-only Transformer

**Framework:** Hugging Face Transformers

**Environment:** Google Colab with NVIDIA T4 GPU

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Hub
* BitsAndBytes
* Accelerate
* Google Colab

---

## Model Loading and Quantization

The LLaMA 3 8B model is loaded using **4-bit quantization** through BitsAndBytes.

The configuration uses:

* 4-bit loading
* NF4 quantization
* Double quantization
* FP16 computation
* Automatic device mapping

4-bit quantization was used because loading the full 8B parameter model exceeded the available Colab GPU memory.

---

## Decoding Strategies

### 1. Greedy Decoding

Greedy decoding selects the token with the highest probability at every generation step.

```text
do_sample = False
```

**Characteristics:**

* Deterministic
* Reproducible
* Less diverse
* Suitable when predictable output is preferred

---

### 2. Temperature Sampling

Temperature modifies the probability distribution before sampling.

```text
do_sample = True
temperature = 1.0
```

A lower temperature generally makes the output more focused and predictable, while a higher temperature increases randomness and diversity.

---

### 3. Top-k Sampling

Top-k sampling restricts the candidate tokens to the **k most probable tokens** before sampling.

Example used in this practical:

```text
do_sample = True
temperature = 0.8
top_k = 50
```

This prevents very low-probability tokens from being selected while still allowing some randomness.

---

### 4. Top-p (Nucleus) Sampling

Top-p sampling dynamically selects the smallest group of tokens whose cumulative probability reaches a specified threshold.

Example used:

```text
do_sample = True
temperature = 0.8
top_p = 0.90
```

Unlike top-k, the number of candidate tokens can change at every generation step depending on the probability distribution.

---

## Comparison of Decoding Strategies

| Technique           | How it works                                                                     | Main effect                                     | Diversity        | Determinism                                                  | How it complements the others                                                                      |
| ------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------- | ---------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **Greedy Decoding** | Selects the highest-probability token                                            | Produces the most predictable continuation      | Low              | High                                                         | Provides a deterministic baseline for comparing sampling-based methods                             |
| **Temperature**     | Adjusts the sharpness of the probability distribution                            | Controls how conservative or random sampling is | Adjustable       | Lower with lower temperature; higher with higher temperature | Can be combined with top-k or top-p to control randomness within a restricted candidate set        |
| **Top-k**           | Samples only from the k highest-probability tokens                               | Removes unlikely token choices                  | Moderate to high | No                                                           | Can be combined with temperature to control both candidate restriction and randomness              |
| **Top-p**           | Samples from the smallest probability set whose cumulative probability reaches p | Dynamically controls the candidate set          | Moderate to high | No                                                           | Can be combined with temperature to control randomness while dynamically filtering unlikely tokens |

### How They Work Together

These techniques control different aspects of generation:

```text
Greedy
   ↓
Choose the single most probable token
   ↓
Deterministic generation

Temperature
   ↓
Adjust probability distribution
   ↓
Control randomness

Top-k
   ↓
Restrict candidates to k probable tokens
   ↓
Avoid extremely unlikely choices

Top-p
   ↓
Restrict candidates using cumulative probability
   ↓
Adapt candidate size to the probability distribution
```

**Temperature + Top-k** and **Temperature + Top-p** are commonly used together. Temperature controls the randomness of the sampling process, while top-k/top-p restrict which tokens are allowed to participate in that sampling process.

---

## Practical Prompt

The same prompt was used to compare the generation strategies:

> What do you mean by Artificial Intelligence? Explain it in brief.

Using the same prompt makes it easier to observe how the decoding strategy affects the generated response.

---

## Key Observations

* **Greedy decoding** provides deterministic and predictable output.
* **Temperature** directly controls the randomness of sampling.
* **Top-k** limits sampling to a fixed number of high-probability candidates.
* **Top-p** dynamically determines the candidate set based on cumulative probability.
* Temperature can be combined with top-k or top-p for more controlled sampling.
* Different decoding configurations can produce different continuations from the same model and prompt.

---

## Important Implementation Detail

The model input tensors are explicitly moved to the GPU before generation:

```python
inputs["input_ids"] = inputs["input_ids"].to("cuda")
inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
```

Generated token IDs are then converted back into readable text using the tokenizer:

```python
result = tokenizer.decode(output[0], skip_special_tokens=True)
```

---

## What I Learned

Through this practical, I implemented LLaMA 3 text generation using Hugging Face and gained hands-on experience with:

* Loading a gated LLM from Hugging Face
* GPU-based model inference
* 4-bit quantization
* BitsAndBytes
* Tokenization
* Causal language model generation
* Greedy decoding
* Temperature sampling
* Top-k sampling
* Top-p sampling
* Comparing generation behavior across decoding strategies

---

## Conclusion

This practical demonstrates how the **same LLaMA 3 model can produce different generation behavior depending on the decoding strategy**.

Greedy decoding provides deterministic generation, while temperature, top-k, and top-p introduce controlled sampling and diversity. Combining temperature with top-k or top-p provides additional control over the generation process.

This practical completes the hands-on **LLaMA text generation and decoding** component of the Transformer learning phase.

---

## Requirements

```text
transformers==4.50.0
accelerate
bitsandbytes
torch
huggingface_hub
```

## Notes

* A Hugging Face account with access to the Meta Llama model is required.
* The Hugging Face access token should be stored securely in Google Colab Secrets.
* The notebook uses 4-bit quantization to make LLaMA 3 8B inference feasible on a Colab T4 GPU.
* Do not hard-code or upload your Hugging Face token to GitHub.

# FLAN-T5 Practical

A practical implementation of **FLAN-T5 (Text-to-Text Transfer Transformer)** using TensorFlow and Hugging Face Transformers.

This practical demonstrates how an instruction-tuned T5 model can perform multiple NLP tasks using the same encoder-decoder Transformer architecture.

## 🧠 Model

This practical uses:

```text
google/flan-t5-base
```

FLAN-T5 is an instruction-tuned version of the T5 model.

T5 uses a **text-to-text** framework, where different NLP tasks are represented as:

```text
Input Text → Model → Output Text
```

For example:

```text
Input:
Classify the sentiment:
I absolutely loved this movie.

Output:
positive
```

## 🛠️ Technologies Used

* Python
* TensorFlow
* Hugging Face Transformers
* FLAN-T5

## 📌 Practical Tasks

### 1. Text Summarization

The model receives a longer piece of text and generates a shorter version.

Example:

```text
Input:
Artificial intelligence is transforming many industries...
Machine learning enables computers to learn patterns from data...
Generative AI allows models to generate text, images, audio, and code...

Output:
Generative AI and machine learning are transforming modern computing.
```

### 2. Translation

The model translates English text into German.

Example:

```text
Input:
Translate the following English sentence into German:
Artificial intelligence is changing the way people work.

Output:
Künstliche Intelligenz verändert die Art und Weise,
wie Menschen arbeiten.
```

### 3. Sentiment Analysis

The model classifies the sentiment of a given sentence.

Example:

```text
Input:
Classify the sentiment:
I absolutely loved this movie. It was amazing.

Output:
positive
```

### 4. Question Answering

The model receives a context and question and generates an answer based on the provided information.

Example:

```text
Context:
Python is a popular programming language used in artificial
intelligence, machine learning, data science, and automation.

Question:
What is Python commonly used for?

Output:
Artificial intelligence, machine learning, data science,
and automation.
```

## 🔄 Text-to-Text Approach

The same FLAN-T5 model can handle different NLP tasks by changing the instruction given to the model.

```text
Summarization
       ↓
"Summarize the following text: ..."

Translation
       ↓
"Translate the following English sentence
```

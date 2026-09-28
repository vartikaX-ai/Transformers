# T5 / FLAN-T5 Practical

A practical implementation of **T5 (Text-to-Text Transfer Transformer)** and **FLAN-T5** using TensorFlow and Hugging Face Transformers.

This practical demonstrates the **text-to-text framework**, instruction-based NLP with FLAN-T5, and **fine-tuning a pretrained T5 model for sentiment classification**.

## 🧠 Models

### T5

T5, or **Text-to-Text Transfer Transformer**, is an encoder-decoder Transformer model that formulates different NLP tasks as text-to-text problems.

Instead of using a separate architecture for each task, T5 follows:

```text
Input Text → Encoder-Decoder Transformer → Output Text
```

For example:

```text
Input:
translate English to German: Hello, how are you?

Output:
Hallo, wie geht es dir?
```

T5 supports tasks such as:

* Translation
* Summarization
* Question answering
* Text classification
* Other text generation tasks

T5 uses task-specific prefixes to indicate the desired task.

### FLAN-T5

The multi-task inference part of this practical uses:

```text
google/flan-t5-base
```

FLAN-T5 is an instruction-tuned version of T5 designed to better follow natural-language instructions.

Example:

```text
Input:
Classify the sentiment:
I absolutely loved this movie.

Output:
positive
```

### Fine-Tuned T5

For the fine-tuning experiment, a smaller T5 checkpoint is used:

```text
t5-small
```

T5-small contains approximately **60 million parameters**, making it considerably more suitable for resource-limited fine-tuning than larger T5 checkpoints.

The fine-tuning task is:

```text
Movie Review → Sentiment
```

Example:

```text
Input:
sentiment: I absolutely loved this movie.

Output:
positive
```

## 🛠️ Technologies Used

* Python
* TensorFlow
* Hugging Face Transformers
* Pandas
* T5
* FLAN-T5
* IMDB Dataset

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
Python is a popular programming language used in
artificial intelligence, machine learning, data science,
and automation.

Question:
What is Python commonly used for?

Output:
Artificial intelligence, machine learning,
data science, and automation.
```

## 🔄 Text-to-Text Approach

The same T5/FLAN-T5 framework can handle different NLP tasks by changing the task instruction or prefix.

```text
Summarization
       ↓
"Summarize the following text: ..."

Translation
       ↓
"Translate the following English sentence into German: ..."

Sentiment Analysis
       ↓
"Classify the sentiment: ..."

Question Answering
       ↓
"Answer the following question using the given context: ..."
```

The model continues to use the same encoder-decoder architecture while the task is expressed through the input text.

---

# 🎯 Fine-Tuning T5 for Sentiment Analysis

This practical also demonstrates **fine-tuning a pretrained T5 model on a downstream NLP task**.

The objective is to adapt T5 so that it learns to generate a sentiment label from a movie review.

```text
Pretrained T5
      ↓
IMDB Movie Reviews
      ↓
Fine-Tuning
      ↓
Sentiment Classification
```

## 📊 Dataset

The **IMDB Dataset** is used for the fine-tuning experiment.

The dataset contains two columns:

```text
review
sentiment
```

The sentiment labels are:

```text
positive
negative
```

A balanced subset was selected for the practical:

```text
Positive reviews: 2500
Negative reviews: 2500
Total reviews:    5000
```

This reduced dataset was used to make the fine-tuning experiment more manageable in the available computing environment.

## 📝 Task Formatting

Because T5 follows a text-to-text framework, sentiment classification is represented as a text generation task.

The input is formatted with a task prefix:

```text
sentiment:
```

Example:

```text
Input:
sentiment: This movie was absolutely fantastic.

Target:
positive
```

Another example:

```text
Input:
sentiment: This movie was terrible and boring.

Target:
negative
```

The model therefore learns:

```text
Review → Generated Sentiment Label
```

## 🔤 Tokenization

The reviews and sentiment labels are tokenized using the T5 tokenizer.

For the practical, the input reviews are limited to a maximum sequence length of **256 tokens**.

The sentiment targets use a much smaller maximum length because the target is only a short label such as:

```text
positive
```

or:

```text
negative
```

This reduces unnecessary memory usage during fine-tuning.

## 🎯 Label Padding

T5 already provides a dedicated padding token.

During training, padding positions in the target labels are replaced with:

```text
-100
```

This causes those positions to be ignored when calculating the training loss.

Conceptually:

```text
<pad> → -100
```

Only the actual target tokens contribute to the loss.

T5 training uses input sequences and corresponding target sequences, with teacher forcing used during training.

## 🔀 Dataset Shuffling

The training examples are shuffled before batching.

```text
IMDB Dataset
      ↓
Balanced subset
      ↓
Shuffle
      ↓
Batch
      ↓
T5
```

Shuffling prevents the model from receiving all examples from one sentiment class consecutively.

This is particularly important when creating a balanced subset by grouping examples according to their sentiment.

## ⚙️ Fine-Tuning Configuration

The practical uses:

```text
Model:
t5-small

Training examples:
5,000

Positive:
2,500

Negative:
2,500

Input maximum length:
256 tokens

Target maximum length:
10 tokens

Batch size:
2

Epochs:
2
```

The smaller `t5-small` checkpoint was selected to make full fine-tuning more practical with limited GPU memory. T5-small is approximately a 60M-parameter model.

## 🧠 Fine-Tuning Process

The complete fine-tuning workflow is:

```text
IMDB Dataset
      ↓
Select balanced subset
      ↓
Create sentiment prompts
      ↓
Tokenize inputs
      ↓
Tokenize target labels
      ↓
Mask padding labels with -100
      ↓
Create TensorFlow Dataset
      ↓
Shuffle and batch
      ↓
Fine-tune pretrained T5
      ↓
Generate sentiment labels
```

## 🧪 Sentiment Prediction

After fine-tuning, new reviews can be passed to the model using the same task prefix.

Example:

```text
Input:
sentiment: This movie was fantastic and amazing.

Output:
positive
```

Example:

```text
Input:
sentiment: I hated this movie. It was awful.

Output:
negative
```

The sentiment is generated as text rather than being produced by a traditional classification head.

## 🔁 Transfer Learning

The fine-tuning experiment demonstrates **transfer learning**.

Instead of training a Transformer from scratch:

```text
Randomly initialized model
        ↓
Train from scratch
```

a pretrained T5 model is reused:

```text
Pretrained T5
      ↓
Task-specific IMDB data
      ↓
Fine-tuning
      ↓
Task-adapted T5
```

The pretrained model's weights are updated using task-specific examples.

## 🧩 T5 Special Tokens

T5 uses several special tokens as part of its architecture and pretraining process.

Important examples include:

```text
<pad>
<eos>
<extra_id_0>
<extra_id_1>
...
```

The `<extra_id_*>` tokens are **sentinel tokens** used during T5's span-corruption pretraining objective.

For example:

```text
Input:
The cat <extra_id_0> the mat.

Target:
<extra_id_0> sat on
```

The sentinel tokens identify masked spans rather than representing the missing text themselves.

## 📚 Concepts Demonstrated

* T5 architecture
* Encoder-decoder Transformer
* Text-to-text framework
* FLAN-T5 instruction tuning
* Task prefixes
* Text summarization
* Machine translation
* Sentiment analysis
* Question answering
* Transfer learning
* Fine-tuning
* Tokenization
* Attention masks
* Padding
* Label masking with `-100`
* TensorFlow datasets
* Dataset shuffling
* Teacher forcing
* Text generation
* Sentinel tokens
* Hugging Face Transformers

## ⚠️ Limitations

* Generated text can occasionally contain information that is not explicitly present in the input.
* Generation quality depends on the model checkpoint and prompt.
* Fine-tuning performance depends on the dataset, training configuration, and available computational resources.
* The sentiment fine-tuning experiment uses a 5,000-review subset rather than the complete IMDB dataset.
* The fine-tuning experiment is intended as a learning and portfolio practical rather than a production sentiment-analysis system.
* No Streamlit interface or production deployment is included.

## 🚀 Conclusion

This practical demonstrates two major ways of using the T5 family.

### Instruction-Based NLP

```text
FLAN-T5
   ↓
Natural-language instruction
   ↓
Multiple NLP tasks
   ↓
Summarization
Translation
Sentiment Analysis
Question Answering
```

### Task-Specific Fine-Tuning

```text
T5
 ↓
Pretrained knowledge
 ↓
IMDB sentiment examples
 ↓
Fine-tuning
 ↓
Sentiment generation
```

Overall, the practical demonstrates how the **T5 text-to-text framework**, **instruction tuning**, and **transfer learning through fine-tuning** can be used to build different NLP capabilities with an encoder-decoder Transformer architecture.

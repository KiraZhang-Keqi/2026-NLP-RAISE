# AI Narrative Intelligence: Multi-Label NLP Pipeline

Turning large-scale news data into structured signals that quantify how AI is framed across economic, cognitive, and social dimensions.

## Overview

This project builds an end-to-end NLP system to analyze **10,500+ news headlines** and map them into a structured **12-category behavioral taxonomy**.

The goal is to move from unstructured text to **quantifiable insights about AI’s societal impact**.

## Key Insights

### Model Performance

| Model                        | Micro-F1 | Improvement |
| ---------------------------- | -------- | ----------- |
| TF-IDF + Logistic Regression | 0.943    | Best        |
| DistilBERT (fine-tuned)      | 0.921    | -2.16pp     |

A key finding is that **classical linear models outperform transformer-based models in short-text classification**, suggesting that sparse representations remain highly effective in high-signal regimes.

### Behavioral Signal Structure

* Categories with explicit lexical markers (e.g., jobs, automation) achieve near-perfect precision (F1 ≈ 0.98)
* Abstract categories (e.g., cognition, emotion) show semantic overlap and reduced separability
* Media narratives exhibit structured patterns but contain inherent ambiguity

### LLM Narrative Analysis

We compare outputs from multiple LLMs (Llama, Mistral, Qwen):

* 55–60% topic overlap across models
* Statistically significant differences in label distributions (χ² test, p < 0.05)

This suggests that while models converge on similar themes, they differ in how they frame narratives.

## Project Significance

### Structured Taxonomy Design

We introduce a 12-category framework spanning:

* Economic impact
* Cognitive processes
* Social dynamics

This enables consistent and scalable analysis of AI-related narratives.


### Practical ML Insight

Rather than defaulting to deep learning, this project:

* Benchmarks classical and transformer-based approaches
* Identifies conditions where simpler models outperform
* Emphasizes model selection based on data characteristics

### Beyond NLP

The pipeline extends into:

* Topic modeling (NMF)
* Financial signal exploration (market correlation)

Linking narrative patterns to broader system-level signals.

## Methodology

```bash
Data → Preprocessing → Multi-label Encoding
      ↓
TF-IDF + Logistic Regression
      ↓
DistilBERT Fine-tuning
      ↓
Evaluation & Error Analysis
      ↓
Topic Modeling + LLM Analysis
```

## Tech Stack

* NLP / ML: scikit-learn, PyTorch, HuggingFace Transformers
* Data: pandas, numpy
* Modeling: Logistic Regression, DistilBERT, NMF
* Statistics: scipy
* Finance: yfinance, arch

## Setup

```bash
pip install pandas numpy scikit-learn transformers torch nltk
pip install yfinance arch
```

Run:

```bash
jupyter notebook NLP_pipeline.ipynb
```

## Project Structure

```bash
.
├── data/
├── notebooks/
│   └── NLP_pipeline.ipynb
├── models/
├── results/
└── README.md
```

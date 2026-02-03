# RAISE-26 AI Behavioral Impact Analysis: A Multi-Label NLP Text Classification Pipeline
An end-to-end NLP pipeline for classifying news headlines into 12 behavioral impact categories, with comparative analysis of LLM-generated content and financial market correlations.

## Project Overview

This project analyzes **10,500+ news headlines** to understand how media frames AI's behavioral impact on society. The pipeline includes:

- **Multi-label text classification** with 12 behavioral categories
- **Model comparison**: TF-IDF + Logistic Regression vs. Fine-tuned DistilBERT
- **LLM narrative analysis**: Comparing outputs from Llama, Mistral, and Qwen
- **Topic modeling** with NMF for thematic discovery
- **Quantitative finance extension**: Correlating media sentiment with market movements

## Key Results

### Classification Performance

| Model | Micro-F1 | Macro-F1 | Weighted-F1 | Samples-F1 |
|-------|----------|----------|-------------|------------|
| **TF-IDF + Logistic Regression** | **0.9430** | **0.9331** | **0.9425** | **0.9216** |
| DistilBERT (Fine-tuned) | 0.9214 | 0.8862 | 0.9149 | 0.9115 |

> The traditional baseline outperformed the transformer model by 2.16pp (Micro-F1), demonstrating that sparse features with linear classifiers can be highly effective for short-text classification.

### Per-Label Performance (Top & Bottom)

| Category | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| Work, Jobs & Economy | 1.000 | 0.965 | **0.982** |
| Learning, Knowledge & Education | 0.984 | 0.959 | 0.971 |
| Technology & Interaction | 0.990 | 0.936 | 0.962 |
| ... | ... | ... | ... |
| Emotion, Motivation & Well-being | 1.000 | 0.800 | 0.889 |
| Cognitive & Decision-Making | 0.807 | 0.893 | 0.848 |

> Categories with clear lexical markers (e.g., "jobs", "automation") achieve near-perfect precision, while abstract concepts (e.g., cognitive, emotional) show semantic overlap with adjacent categories.

### Behavioral Categories

The 12-class taxonomy covers:

| Economic | Cognitive | Social |
|----------|-----------|--------|
| Work, Jobs & Economy | Cognitive & Decision-Making | Social Interaction & Relationships |
| Learning, Knowledge & Education | Creativity, Expression & Identity | Human Roles |
| Technology & Interaction | Emotion, Motivation & Well-being | Society, Ethics & Culture |
| Health, Safety & Risk | Sentiment (Positive/Negative) | Routine, Lifestyle & Behavior |

### Topic Modeling (NMF)

Discovered 10 latent topics from the corpus:

| Topic | Top Keywords |
|-------|--------------|
| Topic 0 | artificial intelligence, stock, prediction, technology |
| Topic 1 | ai innovation, future, data governance, global tech |
| Topic 2 | healthcare, medical diagnosis, patient care |
| Topic 3 | jobs, workers, automation, employment |
| Topic 4 | education, students, learning, classroom |

### LLM Comparison Insights

| LLM | Dominant Categories | Topic Diversity (Entropy) |
|-----|---------------------|---------------------------|
| **Llama** | 7/12 categories (broadest) | High |
| **Mistral** | 3/12 categories | Highest |
| **Qwen** | 2/12 categories (focused) | Lower |

> Despite different architectures, all three LLMs show ~55-60% cluster overlap, suggesting convergent narrative framing. Chi-Square tests confirm significant differences in label distributions across LLMs (p < 0.05). All experiments seeded (SEED=42) with deterministic CUDA operations for reproducibility.

### Statistical Validation

- **Chi-Square Test**: Significant differences in label distributions across LLMs (p < 0.05)
- **Label Distribution**: Maintained consistent across train/val/test splits via stratified sampling
- **Reproducibility**: All experiments seeded (SEED=42) with deterministic CUDA operations

## Pipeline Architecture

```
Data Loading → Preprocessing → Multi-Label Encoding → Train/Val/Test Split
                                                              ↓
                              ┌─────────────────────────────────────────────┐
                              │                                             │
                              ▼                                             ▼
                    TF-IDF + LogReg                              DistilBERT Fine-tuning
                    (Baseline Model)                             (Deep Learning Model)
                              │                                             │
                              └──────────────┬──────────────────────────────┘
                                             ▼
                                   Model Evaluation & Comparison
                                             ↓
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
             Topic Modeling           LLM Output Analysis      Quant Finance Extension
               (NMF)                (Llama/Mistral/Qwen)       (Market Correlation)
```

## Tech Stack

**Core ML/NLP:**
- scikit-learn (TF-IDF, Logistic Regression, NMF)
- PyTorch + HuggingFace Transformers (DistilBERT)
- NLTK (Text preprocessing)

**Analysis & Visualization:**
- pandas, numpy
- matplotlib, seaborn
- scipy (Statistical testing)

**Finance Extension:**
- yfinance, arch (GARCH modeling)

## Environment

#### Run on Google Colab (Recommended)
```bash
# If running in Google Colab, uncomment the following lines.
!pip -q install -U nltk iterative-stratification
!pip -q install -U transformers datasets accelerate

# Local Environment Setup
pip install pandas numpy matplotlib seaborn nltk scikit-learn transformers torch accelerate iterative-stratification

#Quant Analysis Dependencies
!pip install -q yfinance arch
```

## Files

| File | Description |
|------|-------------|
| `RAISE-26 Competition Guidelines.pdf` | Project Guidelines |
| `RAISE-26 NLP Text Classification Pipeline.ipynb` | Complete Analysis Pipeline |
| `RAISE-26 Presentation.pptx` | Project Presentation Slides |
| `RAISE-26 Methodological Analysis.pdf` | Project Theoretical Analysis |
## Acknowledgments

Built for **RAISE 2026** Research Competition.

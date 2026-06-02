# AI Narrative Intelligence

🔗 **Live Demo:** https://2026-nlp-raise-elmt8eifyab9huiuoztk3j.streamlit.app/

An NLP system that transforms **10,500+ AI news headlines** into structured behavioral signals across **12 economic, cognitive, and social categories**.

## Dashboard Preview

![Dashboard](images/dashboard.png)

## Project Overview

AI-related news shapes public perception, investment decisions, and policy discussions, yet most narratives remain unstructured.

This project builds an end-to-end NLP pipeline that classifies AI news headlines into a behavioral taxonomy, enabling large-scale analysis of how AI is framed in public discourse.

The system combines:

* Multi-label text classification
* Topic modeling (NMF)
* Statistical analysis
* LLM narrative comparison
* Interactive dashboard deployment

## Key Results

| Model                        | Micro-F1  |
| ---------------------------- | --------- |
| TF-IDF + Logistic Regression | **0.943** |
| DistilBERT (Fine-Tuned)      | 0.921     |

### Findings

* Classical ML outperformed DistilBERT on short-text classification.
* Categories with explicit keywords (e.g., jobs, automation) achieved the highest accuracy.
* Abstract concepts (e.g., cognition, emotion) showed greater semantic overlap.
* Llama, Mistral, and Qwen converged on similar themes but differed in narrative framing.

## Example Prediction

**Input**

```text
OpenAI launches autonomous AI agents for enterprise workflows
```

**Output**

```text
Automation
Innovation
Productivity
Economic Impact
```

## Dashboard Features

### Headline Classification

Predict behavioral categories for unseen AI news headlines.

![Classification](images/classification.png)

### Narrative Analytics

Explore category distributions, model performance, and latent narrative themes.

![Analytics](images/analytics.png)

### LLM Narrative Comparison

Compare how different open-weight LLMs frame the same AI-related headline.

![LLM Comparison](images/llm_comparison.png)

## Tech Stack

**Machine Learning & NLP**

* Python
* scikit-learn
* PyTorch
* HuggingFace Transformers

**Data Analysis**

* pandas
* NumPy
* SciPy

**Modeling**

* Logistic Regression
* DistilBERT
* NMF Topic Modeling

**Deployment**

* Streamlit
* GitHub

## Repository Structure

```text
.
├── README.md
├── app.py
├── NLP_Pipeline.ipynb
├── Project_Presentation.pdf
├── requirements.txt
└── images/
```

## Skills Demonstrated

* Multi-Label NLP
* Text Classification
* Feature Engineering
* Model Evaluation
* Topic Modeling
* Statistical Testing
* Data Visualization
* Dashboard Development

---


# RAGShield — Indirect Prompt Injection Detection in RAG Systems

**Doctoral Praxis — George Washington University**
**Author: T. Nongni**
**Repository: https://github.com/nongnited/DoctoralPraxis**

---

## Overview

RAGShield is a security layer that detects and blocks indirect prompt injection attacks in
Retrieval-Augmented Generation (RAG) systems. It intercepts retrieved documents between the
retrieval and generation layers, classifying each document before it reaches the LLM context window.

This repository contains the complete reproducible pipeline for the doctoral praxis:

1. Dataset download and preprocessing
2. Feature engineering (TF-IDF, MiniLM, mpnet)
3. Classifier training (10 models × 3 feature sets × 3 balancing strategies)
4. Deployment (Docker containers, FastAPI gateway, ChromaDB, Ollama)
5. Evaluation corpus generation and upload
6. Pipeline evaluation (4 injection rate scenarios × 10 models)
7. Results analysis (MCC, F1, FAR, latency)

---

## Infrastructure Requirements

### Training Machine
- Apple MacBook Pro (2024) or equivalent with 32GB+ RAM
- Python 3.10
- macOS or Linux

### Deployment Cluster (5 servers)
| Server   | Type                  | Role               | Requirements                        |
|----------|-----------------------|--------------------|-------------------------------------|
| beacon   | Proxmox VM            | DNS                | 4 cores, 8GB RAM, Rocky Linux 9     |
| vault    | Proxmox VM            | RAG Gateway        | 8 cores, 32GB RAM, Rocky Linux 9    |
| shield   | Bare metal (HP DL380) | Classifier Server  | 16+ cores, 64GB+ RAM, Rocky Linux 9 |
| arbiter  | Bare metal (Dell R740)| LLM Inference      | Dual P100 GPU, 256GB RAM            |
| ledger   | Proxmox VM            | Database           | 8 cores, 32GB RAM, Rocky Linux 9    |

---

## Quickstart

### Step 1 — Clone and install dependencies
```bash
git clone https://github.com/nongnited/DoctoralPraxis.git
cd DoctoralPraxis
pip install -r requirements.txt
```

### Step 2 — Configure environment
```bash
cp .env.example .env
# Edit .env with your server IPs and credentials
```

### Step 3 — Download and preprocess datasets
```bash
python 1_data/download_datasets.py
python 1_data/preprocess_injection.py
python 1_data/preprocess_benign.py
python 1_data/build_corpus.py
```

### Step 4 — Extract features
```bash
python 2_features/extract_tfidf.py
python 2_features/extract_minilm.py
python 2_features/extract_mpnet.py
```

### Step 5 — Train all models
```bash
bash 3_training/train_all.sh
```
> Warning: Full training takes 12-24 hours. BERT models require GPU.
> Individual models can be trained separately — see `3_training/README.md`

### Step 6 — Deploy infrastructure
```bash
# On shield server
cd 4_deployment && docker compose up -d

# On vault server
cd 4_deployment/vault && docker compose up -d
```

### Step 7 — Upload evaluation corpus
```bash
# Corpus (500 documents) is included in 5_corpus/corpus/
python 5_corpus/upload_corpus.py
```

### Step 8 — Run evaluation
```bash
# Run all 4 scenarios × 10 models
for scenario in s1_10pct s2_20pct s3_30pct s4_50pct; do
    python -u 6_evaluation/scenarios/$scenario/run_all_models.py \
        2>&1 | tee 6_evaluation/scenarios/$scenario/results/run_$(date +%Y%m%d_%H%M).txt
done
```

### Step 9 — Analyze results
```bash
python 7_analysis/analyze_training_results.py
python 7_analysis/analyze_latency.py
```

---

## Repository Structure

```
RAGShield/
├── README.md
├── requirements.txt
├── .env.example
│
├── 1_data/                    # Dataset download and preprocessing
├── 2_features/                # Feature extraction (TF-IDF, MiniLM, mpnet)
├── 3_training/                # Model training scripts (10 classifiers)
├── 4_deployment/              # Docker containers and FastAPI services
├── 5_corpus/                  # Evaluation corpus + upload scripts
├── 6_evaluation/              # Pipeline evaluation scripts
├── 7_analysis/                # Results analysis scripts
└── infrastructure/            # Database schema, server setup guides
```

---

## Reproducing Trained Models

Trained model weights are not distributed with this repository.
All models must be retrained from scratch using the pipeline above (Steps 1–5).
Every hyperparameter needed to reproduce the published results is documented
in the training scripts and summarised here:

| Model               | Key Hyperparameters                                          |
|---------------------|--------------------------------------------------------------|
| Naïve Bayes         | MultinomialNB (TF-IDF), GaussianNB (dense)                  |
| Logistic Regression | saga solver, L2, C=1.0, max_iter=2000                       |
| Ridge               | alpha=1.0, copy_X=False                                      |
| Random Forest       | 300 trees, max_features=sqrt, n_jobs=-1                      |
| Extra Trees         | 300 trees, max_features=sqrt, n_jobs=-1                      |
| XGBoost             | hist, 500 trees, lr=0.1, subsample=0.8, auto CUDA           |
| LightGBM            | 500 trees, 127 leaves, early_stopping=50, auto GPU          |
| SVM                 | LinearSVC (TF-IDF) / SGDClassifier-hinge (dense) + scaler   |
| BERT Base           | bert-base-uncased, 3 epochs, lr=2e-5, batch=16, max_len=128 |
| SecureBERT          | cisco-ai/SecureBERT2.0-base, same as BERT, use_fast=False   |

**Feature sets per classical model:** TF-IDF (50K dims, unigrams+bigrams),
MiniLM (384-dim), mpnet (768-dim).
**Balancing strategies:** none, class_weight=balanced, SMOTE (dense only).
**Total training runs:** 64 classical + 2 BERT = 66 configurations.

Training saves models automatically to `models/` in the structure expected
by the deployment containers. No manual file placement required.

---

## Training Datasets

| Dataset      | Size     | Source                                      |
|--------------|----------|---------------------------------------------|
| BIPIA        | ~200K    | microsoft/BIPIA (Hugging Face)              |
| LLMail-Inject| ~173K    | Custom download (see 1_data/download.py)    |
| TensorTrust  | ~160K    | HuggingFace datasets                        |
| Benign       | ~1.3M    | Enron, arXiv, Wikipedia, legal, financial   |
| **Total**    | **1.63M**| 80/20 train/val split                       |

---

## Citation

If you use this code or the RAGShield evaluation methodology in your research, please cite:

```
Nongni, T. (2026). RAGShield: Indirect Prompt Injection Detection in
Retrieval-Augmented Generation Systems. Doctoral Praxis,
George Washington University.
```

---

## License

This repository is provided for academic reproducibility purposes.
All third-party datasets are subject to their respective licenses.

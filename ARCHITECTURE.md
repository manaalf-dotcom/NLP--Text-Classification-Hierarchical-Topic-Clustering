# Architecture & Data Flow

## Overview
The pipeline processes raw text documents through three stages:
classification with classic features, classification with semantic
embeddings, and unsupervised topic clustering.

## Data Flow
```
Raw Text (20 Newsgroups, 18,846 docs)
        │
        ▼
  Train/Test Split (80/20, stratified)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
   PART 1                                 PART 2
   TF-IDF Vectorizer                      SentenceTransformer
   (50,000 features)                      (384-dim embeddings)
        │                                      │
        ▼                                      ▼
   sklearn Pipeline                       4 Classifiers
   (vectorizer + model)                   (NB, LR, SVM, RF)
        │                                      │
        ▼                                      ▼
   4 Classifiers                          Accuracy + Macro F1
   (NB, LR, SVM, RF)                           │
        │                                      │
        ▼                                      ▼
   Accuracy + Macro F1              PART 3 (uses Part 2 embeddings)
   Confusion Matrix                            │
                                               ▼
                                    KMeans Clustering (K=6)
                                    Elbow Method to pick K
                                               │
                                               ▼
                                    Find centroid documents
                                               │
                                               ▼
                                    Gemini LLM → Topic Labels
                                               │
                                               ▼
                                    Sub-cluster 2 biggest clusters
                                    into 3 sub-topics each
                                               │
                                               ▼
                                    2-Level Topic Tree
```

## Module Responsibilities

### Part 1 — Classic Features
- `fetch_20newsgroups` loads and cleans raw text
- `train_test_split` creates 80/20 stratified split
- `TfidfVectorizer` converts text to numerical features
- `Pipeline` chains vectorizer and classifier to prevent leakage
- Four classifiers trained and evaluated: MNB, LogReg, LinearSVC, RandomForest

### Part 2 — Semantic Embeddings
- `SentenceTransformer('all-MiniLM-L6-v2')` encodes documents into 384-dim vectors
- Same 4 classifiers trained on embeddings
- `MinMaxScaler` applied before Naive Bayes to handle negative values
- Results compared against Part 1

### Part 3 — Clustering & Topic Tree
- `KMeans` clusters documents using SentenceTransformer embeddings
- Elbow method (WCSS plot) used to select optimal K=6
- `NearestNeighbors` finds 5 documents closest to each centroid
- Gemini LLM (`gemini-2.5-flash-lite`) generates 2-4 word topic labels
- 2 largest clusters sub-divided into 3 sub-clusters each
- Topic tree printed showing parent and child cluster labels

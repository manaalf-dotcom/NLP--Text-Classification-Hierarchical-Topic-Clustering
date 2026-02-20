# Text Classification & Hierarchical Topic Clustering

## Overview
An end-to-end NLP pipeline built on the 20 Newsgroups dataset (18,846 documents, 20 categories).
The project compares classic text features vs semantic embeddings for classification, then performs
hierarchical topic clustering with LLM-generated labels.

## Project Structure
- `notebook.ipynb` — Main notebook containing all 3 parts
- `README.md` — Project overview and setup instructions
- `ARCHITECTURE.md` — Data flow and module responsibilities

## Setup
Run the following in your notebook or terminal to install dependencies:
```bash
pip install scikit-learn sentence-transformers google-generativeai matplotlib seaborn pandas numpy
```

## How to Run

### Part 1 — Classic Classification (BoW/TF-IDF)
- Loads 20 Newsgroups dataset
- Trains 4 classifiers using TF-IDF features inside sklearn Pipelines
- Reports Accuracy, Macro F1, Confusion Matrix and top confusion pairs
- Expected best model: Linear SVM (~77% accuracy)

### Part 2 — SentenceTransformer Embeddings
- Encodes all documents using `all-MiniLM-L6-v2` SentenceTransformer model
- Trains same 4 classifiers on dense embeddings
- Compares results against Part 1

### Part 3 — Topic Clustering & Topic Tree
- Uses KMeans clustering on SentenceTransformer embeddings
- Elbow method used to select K=6 clusters
- Gemini LLM generates topic labels for each cluster
- 2 largest clusters are sub-clustered into 3 sub-topics each
- Final 2-level topic tree is displayed

## Example Output

### Part 1 Results
| Model | Accuracy | Macro F1 |
|---|---|---|
| Linear SVM | 0.7682 | 0.7617 |
| Logistic Regression | 0.7374 | 0.7259 |
| Naive Bayes | 0.6966 | 0.6693 |
| Random Forest | 0.6350 | 0.6137 |

### Topic Tree
```
📁 Motorcycle Sales (4773 docs)
    ├── Medical Advice
    ├── Classifieds
    └── Motorcycle Sales

📁 Computer Sales (3990 docs)
    ├── Graphics Software
    ├── Computer Hardware
    └── Computer Equipment Sales

📄 Waco Controversy (2904 docs)
📄 Religious Debate (1624 docs)
📄 Hockey Playoffs (1379 docs)
📄 Empty/Noise Documents (406 docs)
```

## Requirements
- Python 3.8+
- Google Colab (recommended) or local Jupyter
- Gemini API key (free at aistudio.google.com)

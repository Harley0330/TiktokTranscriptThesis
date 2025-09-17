TikTok Fake News Classification using GNN + Random Forest
--------------------------------------------------------

This project implements a fake news detection pipeline for TikTok videos using
the FakeTT dataset (ACM MM 2024). Video transcripts are processed into textual
features, which are then classified using a hybrid Graph Neural Network (GNN)
and Random Forest classifier.

Dataset
-------
- FakeTT Dataset (TikTok video transcripts)
- Columns: video_id, description, transcript, label
- Labels: 1 = Fake news, 0 = Real news

Pipeline Overview
-----------------
1. Data Preprocessing (src/preprocessing.py)
   - Lowercasing
   - Removal of special characters (keeps hashtags + numbers)
   - Tokenization
   - Stopword removal
   - Dropping empty transcripts

2. Text Feature Extraction (src/feature_extraction.py)
   - TF-IDF vectorization (scikit-learn)
   - Word co-occurrence graph construction (NetworkX)

3. Model Training
   - Graph Neural Network (PyTorch Geometric)
   - Random Forest classifier (scikit-learn)
   - Fusion of GNN probability scores with Random Forest features

4. Evaluation
   - Metrics: Accuracy, Precision, Recall, F1-score

How to Run
----------
1. Install dependencies:
   pip install -r requirements.txt

2. Place dataset in:
   data/FakeTT_dataset.csv (for the meantime data/testing.csv)

3. Run preprocessing:
   python main.py

4. (Later) Run feature extraction and model training.

Dependencies
------------
- Python 3.10+ recommended
- pandas
- nltk
- scikit-learn
- torch
- torch-geometric
- networkx

Project Structure
-----------------
FakeNewsClassifier-GNN-RF/
│
├── data/                     # Dataset (raw and processed)
├── src/                      # Source code
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── gnn_model.py
│   ├── random_forest.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── main.py                   # Pipeline entry point
├── requirements.txt
└── README.txt

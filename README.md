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
      - Save Models in models directory
   - Random Forest classifier (scikit-learn)
   - Fusion of GNN probability scores with Random Forest features

4. Evaluation
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Kolmogorov-Smirnov Test
   - Paired T-test / Wilcoxson Signed Rank Test
   - McNemar's Test

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
```bash
TikTokThesis/
├── data/
│   ├── raw/
│   │   └── data_cleaned.csv
│   └── processed/
│       └── data_cleaned_formatted.csv
├── models/
│   ├── gnn_fold1_best.pth
│   ├── gnn_fold2_best.pth
│   ├── gnn_fold3_best.pth
│   ├── gnn_fold4_best.pth
│   ├── gnn_fold5_best.pth
│   ├── gnn_fold6_best.pth
│   ├── gnn_fold7_best.pth
│   ├── gnn_fold8_best.pth
│   ├── gnn_fold9_best.pth
│   ├── gnn_fold10_best.pth
│   ├── gnn_fold11_best.pth
│   ├── gnn_fold12_best.pth
│   ├── gnn_fold13_best.pth
│   ├── gnn_fold14_best.pth
│   ├── gnn_fold15_best.pth
│   ├── gnn_training_log.csv
│   └── rf_final.pkl
├── plots/
│   ├── gnn_plots/                  
│   ├── hybrid_model_plots/         
│   └── hypothesis_testing_plots/   
├── results/
│   ├── baseline_fold_metrics.csv
│   ├── baseline_predictions_full.csv
│   ├── baseline_rf_metrics.csv
│   ├── gnn_training_log.csv
│   ├── hybrid_fold_metrics.csv
│   ├── hybrid_fold_metrics_calibrated.csv
│   ├── hybrid_predictions_full.csv
│   └── hybrid_predictions_calibrated.csv
├── src/
│   ├── __init__.py
│   ├── baseline_hybrid_comparison_plots.py
│   ├── feature_extraction.py
│   ├── gnn_model.py
│   ├── gnn_plots.py
│   ├── hybrid_rf_gnn.py
│   ├── preprocessing.py
│   ├── random_forest.py
│   ├── statistical_tests.py
│   ├── statistical_testing_plots.py
│   ├── train_gnn.py
│   ├── train.py
│   └── utils.py
├── requirements.txt          # Dependencies (torch, torch-geometric, sklearn, nltk, etc.)
├── config.yaml               # Hyperparameters (layers, batch size, RF params, etc.)
├── main.py                   # Entry point: end-to-end pipeline
├── analyzer_app.py           # Entry point: Frontend
├── DESKTOP_APP_SETUP.md      # Guide to setup the Frontend       
└── README.md                 # Documentation
```

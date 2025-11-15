"""
Save models and components needed for the desktop application
Run this after training your models with main.py or train_gnn.py
Place in your project root (same level as main.py) and run: python save_for_api.py
"""

import torch
import joblib
import numpy as np
from pathlib import Path
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Now import from src
from src.preprocessing import preprocess_dataset
from src.train import prepare_data
from src.feature_extraction import build_word_occurrence_graph
from src.utils import RAW_DIR, PROCESSED_DIR, MODELS_DIR
from sklearn.decomposition import TruncatedSVD

def save_components_for_api():
    """Save all components needed for the API"""
    
    print("=" * 60)
    print("Saving components for Desktop Application...")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load and preprocess dataset
    print("\n[1/4] Loading dataset...")
    try:
        df = preprocess_dataset(RAW_DIR / "data_cleaned.csv")
        tokens_list = df["tokens"].tolist()
        print(f"   ✓ Loaded {len(df)} samples")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return
    
    # 2. Build TF-IDF and save vectorizer
    print("[2/4] Building TF-IDF and saving vectorizer...")
    try:
        output_path = PROCESSED_DIR / "data_cleaned_formatted.csv"
        X, y, vectorizer = prepare_data(df, output_path, max_features=5000)
        
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        joblib.dump(vectorizer, vectorizer_path)
        print(f"   ✓ Saved vectorizer to: {vectorizer_path}")
    except Exception as e:
        print(f"   ✗ Error building TF-IDF: {e}")
        return
    
    # 3. Build graph and save structure
    print("[3/4] Building word co-occurrence graph...")
    try:
        vocab = vectorizer.get_feature_names_out()
        vocab_index = {w: i for i, w in enumerate(vocab)}
        vocab_set = set(vocab)
        
        G = build_word_occurrence_graph(
            tokens_list=tokens_list, 
            window_size=2,
            vocab_set=vocab_set
        )
        
        print(f"   Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # Convert TF-IDF to dense
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        num_nodes = len(vocab)
        num_docs = X_dense.shape[0]
        
        # Node features: TF-IDF values per word across all docs
        node_features = np.zeros((num_nodes, num_docs))
        
        for word, idx in vocab_index.items():
            node_features[idx] = X_dense[:, idx]
        
        # Reduce dimensionality
        svd = TruncatedSVD(n_components=64, random_state=42)
        node_features_reduced = svd.fit_transform(node_features)
        
        # Convert to torch
        x = torch.tensor(node_features_reduced, dtype=torch.float, device=device)
        
        # Edges
        edge_index = torch.tensor(
            [[vocab_index[u], vocab_index[v]] for u, v in G.edges() 
             if u in vocab_index and v in vocab_index],
            dtype=torch.long, device=device
        ).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Save graph structure
        graph_data = {
            'vocab_index': vocab_index,
            'x_graph': x.cpu(),  # Save on CPU for compatibility
            'edge_index': edge_index.cpu(),
            'vocab_list': list(vocab)
        }
        
        graph_path = MODELS_DIR / "graph_structure.pkl"
        joblib.dump(graph_data, graph_path)
        print(f"   ✓ Saved graph structure to: {graph_path}")
    except Exception as e:
        print(f"   ✗ Error building graph: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Verify saved models exist
    print("[4/4] Checking for trained models...")
    
    rf_path = MODELS_DIR / "rf_final.pkl"
    if rf_path.exists():
        print(f"   ✓ Random Forest model found: {rf_path}")
    else:
        print(f"   ⚠️  Random Forest model NOT found: {rf_path}")
        print("      Run random_forest.py or hybrid_rf_gnn.py to train it")
    
    gnn_models = list(MODELS_DIR.glob("gnn_fold*_best.pth"))
    if gnn_models:
        print(f"   ✓ Found {len(gnn_models)} GNN fold models")
        for model_path in sorted(gnn_models)[:3]:
            print(f"      - {model_path.name}")
    else:
        print(f"   ⚠️  No GNN models found in {MODELS_DIR}")
        print("      Run train_gnn.py or main.py to train them")
    
    print("\n" + "=" * 60)
    print("✅ Components saved successfully!")
    print("=" * 60)
    print("\nSaved files:")
    print(f"  ✓ {vectorizer_path}")
    print(f"  ✓ {graph_path}")
    print("\nNext steps:")
    print("1. Make sure your models are trained")
    print("2. Run the desktop app: python analyzer_app.py")
    print()

if __name__ == "__main__":
    # Change to project root directory
    os.chdir(project_root)
    save_components_for_api()
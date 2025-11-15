# FakeTT Analyzer - Desktop Application Setup Guide

## 📋 Overview
Standalone desktop application for analyzing TikTok transcripts using your trained GNN + Random Forest model.

## 🗂️ File Structure
```
your-project/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── gnn_model.py
│   ├── random_forest.py
│   ├── train.py
│   ├── hybrid_rf_gnn.py
│   └── utils.py
├── data/
│   └── data_cleaned.csv
├── models/           # Your trained models go here
├── main.py
├── analyzer_app.py   # NEW - Desktop application
└── save_for_api.py   # NEW - Save components for app
```

## 🚀 Quick Start

### Step 1: Train Your Models (if not done yet)
```bash
# Train GNN models (creates gnn_fold*_best.pth files)
python train_gnn.py

# Train hybrid RF+GNN model (creates rf_final.pkl)
python hybrid_rf_gnn.py
```

### Step 2: Save Components for Application
```bash
python save_for_api.py
```

This will save:
- ✅ TF-IDF vectorizer (`tfidf_vectorizer.pkl`)
- ✅ Graph structure (`graph_structure.pkl`)
- ✅ Verifies your trained models exist

### Step 3: Run the Desktop Application
```bash
python analyzer_app.py
```

## 🖥️ Using the Application

### Interface Overview
```
┌─────────────────────────────────────────────────┐
│  FakeTT Analyzer                    ● Ready     │
│  TikTok Fake News Detection                     │
├─────────────────────────────────────────────────┤
│  TikTok URL                                     │
│  ┌─────────────────────────────────────────┐   │
│  │ https://www.tiktok.com/@user/video/... │   │
│  └─────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│  Video Transcript                               │
│  ┌─────────────────────────────────────────┐   │
│  │                                          │   │
│  │  Paste your transcript here...          │   │
│  │                                          │   │
│  └─────────────────────────────────────────┘   │
│                          Characters: 0          │
├─────────────────────────────────────────────────┤
│             [ Analyze Transcript ]              │
├─────────────────────────────────────────────────┤
│  Analysis Results                               │
│                                                  │
│         ✓ LIKELY REAL NEWS                      │
│                                                  │
│  Fake Probability:              45.23%          │
│  [████████░░░░░░░░░░░░░░░░░░░░]                │
│                                                  │
│  Model Confidence:           67.54% (Medium)    │
└─────────────────────────────────────────────────┘
```

### Step-by-Step Usage

1. **Wait for Models to Load**
   - On first launch, the app loads models in the background
   - Status indicator shows: "● Loading models..."
   - Once ready: "● Models loaded - Ready"

2. **Enter TikTok URL** (optional)
   - Paste the TikTok video URL for reference

3. **Paste Transcript**
   - Copy the video transcript
   - Paste into the large text area
   - Character count updates automatically

4. **Click "Analyze Transcript"**
   - Button becomes disabled during analysis
   - Shows "Analyzing..." while processing

5. **View Results**
   - Classification: FAKE NEWS or REAL NEWS
   - Probability bar: Visual representation
   - Confidence level: How certain the model is

## 📊 Understanding Results

### Classification
- **✓ LIKELY REAL NEWS** (Green): Predicted as real/legitimate
- **⚠️ FAKE NEWS DETECTED** (Red): Predicted as fake/misleading

### Fake Probability
- **0-50%**: More likely to be real
- **50-70%**: Uncertain, requires human verification
- **70-100%**: More likely to be fake

### Model Confidence
- **High (80-100%)**: Model is very confident
- **Medium (60-80%)**: Moderate confidence
- **Low (0-60%)**: Low confidence, manual review recommended

### Color Coding
- 🟢 **Green**: Real/Safe (< 50% fake probability)
- 🟡 **Yellow**: Uncertain (50-70% fake probability)
- 🔴 **Red**: Fake/Dangerous (> 70% fake probability)

## 🔧 Technical Details

### Models Used
1. **Graph Neural Network (GNN)**
   - 4-layer GraphSAGE with skip connections
   - Learns word co-occurrence patterns
   - 64-dimensional node embeddings

2. **Random Forest Classifier**
   - 2000 decision trees
   - Balanced class weights
   - Combines TF-IDF + GNN features

### Processing Pipeline
1. Text preprocessing (lowercase, stopwords, tokenization)
2. TF-IDF vectorization (5000 features)
3. Word co-occurrence graph construction
4. GNN embedding extraction
5. Feature fusion and classification

### Performance
- Loads models once at startup (~10-30 seconds)
- Analysis per transcript: ~50-200ms
- GPU acceleration supported (if available)
- Memory usage: ~500MB-1GB

## 🐛 Troubleshooting

### App won't start
```bash
# Check Python version (3.8+ required)
python --version

# Verify tkinter is installed
python -c "import tkinter"

# If missing, install tkinter:
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (usually pre-installed)
brew install python-tk
```

### "Models are still loading"
- Wait 10-30 seconds for initial model loading
- Check console for loading progress
- Look for "✓ Loaded" messages

### "Failed to load models" error
Run the save script first:
```bash
python save_for_api.py
```

Verify these files exist in `models/`:
- ✓ `tfidf_vectorizer.pkl`
- ✓ `graph_structure.pkl`
- ✓ `rf_final.pkl`
- ✓ `gnn_fold1_best.pth`, `gnn_fold2_best.pth`, etc.

### "Transcript is empty after preprocessing"
- Transcript may be too short
- Try adding more text content
- Ensure transcript contains actual words (not just symbols)

### Slow analysis
- First analysis may be slower (model initialization)
- Subsequent analyses are faster
- GPU acceleration helps (if available)
- Very long transcripts (>5000 words) take longer

### Memory issues
- Close other applications
- Use shorter transcripts
- Reduce number of GNN models loaded (modify line 80 in analyzer_app.py)

## 💡 Tips for Best Results

1. **Transcript Quality**
   - Use complete, accurate transcripts
   - Include all spoken content
   - Remove timestamps and metadata

2. **Context Matters**
   - Longer transcripts generally work better
   - Minimum ~50 words recommended
   - More context = better predictions

3. **Interpreting Results**
   - Don't rely solely on the model
   - Use results as a first-pass filter
   - Human verification for high-stakes decisions
   - Consider confidence levels

4. **Batch Processing**
   - For multiple transcripts, run the app multiple times
   - Or modify the code to add batch processing
   - Results can be copied from the UI

## 📝 Keyboard Shortcuts

- **Ctrl+A**: Select all (in transcript area)
- **Ctrl+C**: Copy selected text
- **Ctrl+V**: Paste text
- **Ctrl+X**: Cut selected text

## 🔒 Privacy & Security

- **All processing is local** - no data sent to external servers
- Transcript data is not saved unless you explicitly save it
- Models run entirely on your machine
- No internet connection required (after model training)

## 📈 Performance Metrics

Your model's expected performance:
- Accuracy: ~75-85% (dataset dependent)
- Precision: Check `results/hybrid_fold_metrics.csv`
- Recall: Check training logs
- F1-Score: Balanced measure

## 🎯 Use Cases

1. **Content Moderation**
   - Quick screening of TikTok videos
   - Flagging potentially misleading content

2. **Research**
   - Analyzing misinformation patterns
   - Dataset creation and labeling

3. **Education**
   - Teaching media literacy
   - Demonstrating ML applications

4. **Journalism**
   - Fact-checking assistance
   - Source verification

## 🛠️ Advanced Configuration

### Modify Number of GNN Models
Edit `analyzer_app.py`, line ~80:
```python
for path in model_paths[:5]:  # Change 5 to load more/fewer models
```

### Adjust Confidence Thresholds
Edit display_results method to change color thresholds:
```python
if conf_pct >= 80:  # Adjust these values
    conf_text += " (High)"
```

### GPU/CPU Selection
The app automatically uses GPU if available. To force CPU:
```python
self.device = torch.device("cpu")  # Force CPU in load_models()
```

## 📞 Support

If you encounter issues:
1. Check console output for error messages
2. Verify all files are in correct locations
3. Ensure models are trained properly
4. Check Python and package versions
5. Review troubleshooting section above
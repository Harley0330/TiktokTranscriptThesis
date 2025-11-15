import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import preprocess_dataset
# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.preprocessing import preprocess_pipeline  # Import the preprocessing pipeline
from src.feature_extraction import build_word_occurrence_graph
from src.train import prepare_data
from src.gnn_model import GNNClassifier, extract_gnn_probabilities
from src.utils import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import scipy.sparse as sp
import joblib

class FakeTTAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FakeTT Analyzer - Fake News Detection")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Model variables
        self.rf_model = None
        self.gnn_models = []
        self.df = preprocess_dataset(RAW_DIR/"data_cleaned.csv")
        self.output_path = PROCESSED_DIR / "data_cleaned_formatted.csv"
        self.X, self.y, self.vectorizer = prepare_data(self.df,self.output_path,max_features=5000)
        self.vocab_index = None
        self.x_graph = None
        self.edge_index = None
        self.device = None
        self.models_loaded = False
        self.use_gnn = False
        self.model_type = "tfidf_only"
        self.scalers = None
        
        # Setup UI
        self.setup_ui()
        
        # Load models in background
        self.load_models_async()
    
    def setup_ui(self):
        """Create the user interface"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(4, weight=0)
        
        # ===== HEADER =====
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        header_frame.columnconfigure(0, weight=1)
        
        title_label = tk.Label(
            header_frame, 
            text="FakeTT Analyzer", 
            font=("Helvetica", 24, "bold"),
            fg="#1e40af"
        )
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        subtitle_label = tk.Label(
            header_frame,
            text="TikTok Fake News Detection using GNN + Random Forest",
            font=("Helvetica", 10),
            fg="#64748b"
        )
        subtitle_label.grid(row=1, column=0, sticky=tk.W)
        
        # Status indicator
        self.status_label = tk.Label(
            header_frame,
            text="● Loading models...",
            font=("Helvetica", 9),
            fg="#f59e0b"
        )
        self.status_label.grid(row=0, column=1, sticky=tk.E)
        
        # ===== URL INPUT =====
        url_frame = ttk.LabelFrame(main_frame, text="TikTok URL (Optional)", padding="10")
        url_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        url_frame.columnconfigure(0, weight=1)
        
        url_input_frame = ttk.Frame(url_frame)
        url_input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        url_input_frame.columnconfigure(0, weight=1)
        
        self.url_entry = ttk.Entry(url_input_frame, font=("Helvetica", 10))
        self.url_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        self.url_entry.insert(0, "https://www.tiktok.com/@username/video/...")  # Placeholder text
        self.url_entry.bind("<FocusIn>", self.clear_url_placeholder)
        
        self.fetch_button = ttk.Button(
            url_input_frame,
            text="Fetch Transcript",
            command=self.fetch_transcript_from_url,
            state=tk.DISABLED
        )
        self.fetch_button.grid(row=0, column=1, padx=(0, 5), pady=5)
        
        tk.Label(
            url_frame,
            text="Or enter transcript manually below",
            font=("Helvetica", 8),
            fg="#64748b"
        ).grid(row=1, column=0, pady=(5, 0))
        
        # ===== TRANSCRIPT INPUT =====
        transcript_frame = ttk.LabelFrame(main_frame, text="Video Transcript", padding="10")
        transcript_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        transcript_frame.columnconfigure(0, weight=1)
        transcript_frame.rowconfigure(0, weight=1)
        
        self.transcript_text = scrolledtext.ScrolledText(
            transcript_frame,
            height=12,
            font=("Consolas", 10),
            wrap=tk.WORD,
            bg="#f8fafc",
            relief=tk.SOLID,
            borderwidth=1
        )
        self.transcript_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Character count
        self.char_count_label = tk.Label(
            transcript_frame,
            text="Characters: 0",
            font=("Helvetica", 8),
            fg="#64748b"
        )
        self.char_count_label.grid(row=1, column=0, sticky=tk.E, padx=5)
        
        self.transcript_text.bind("<KeyRelease>", self.update_char_count)
        
        # ===== ANALYZE BUTTON =====
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))
        button_frame.columnconfigure(0, weight=1)
        
        self.analyze_button = ttk.Button(
            button_frame,
            text="Analyze Transcript",
            command=self.analyze_transcript,
            state=tk.DISABLED
        )
        self.analyze_button.pack(pady=5, ipadx=20, ipady=10)
        
        # ===== RESULTS FRAME =====
        results_container = ttk.Frame(main_frame)
        results_container.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_container.columnconfigure(0, weight=1)
        
        self.results_frame = ttk.LabelFrame(results_container, text="Analysis Results", padding="15")
        self.results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_frame.columnconfigure(0, weight=1)
        
        # Initially hidden
        results_container.grid_remove()
        self.results_container = results_container
        
        # Create result widgets
        self.setup_results_widgets()
    
    def setup_results_widgets(self):
        """Setup the results display widgets"""
        # Classification result
        self.result_label = tk.Label(
            self.results_frame,
            text="",
            font=("Helvetica", 18, "bold"),
            fg="#10b981"
        )
        self.result_label.grid(row=0, column=0, pady=(0, 15))
        
        # Probability bar
        prob_frame = ttk.Frame(self.results_frame)
        prob_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        prob_frame.columnconfigure(1, weight=1)
        
        tk.Label(
            prob_frame,
            text="Fake Probability:",
            font=("Helvetica", 10, "bold")
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.prob_label = tk.Label(
            prob_frame,
            text="0.00%",
            font=("Helvetica", 10)
        )
        self.prob_label.grid(row=0, column=2, sticky=tk.E)
        
        # Progress bar for probability
        self.prob_bar = ttk.Progressbar(
            prob_frame,
            length=400,
            mode='determinate'
        )
        self.prob_bar.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Model info
        info_text = "Model: Hybrid GNN + Random Forest with TF-IDF features"
        tk.Label(
            self.results_frame,
            text=info_text,
            font=("Helvetica", 8),
            fg="#64748b"
        ).grid(row=2, column=0, pady=(15, 0))
    
    def clear_url_placeholder(self, event):
        """Clear URL placeholder text on focus"""
        if self.url_entry.get().startswith("https://www.tiktok.com/@username"):
            self.url_entry.delete(0, tk.END)
    
    def update_char_count(self, event=None):
        """Update character count label"""
        text = self.transcript_text.get("1.0", tk.END).strip()
        count = len(text)
        self.char_count_label.config(text=f"Characters: {count}")
    
    def fetch_transcript_from_url(self):
        """Fetch transcript from TikTok URL"""
        url = self.url_entry.get().strip()
        
        if not url or url.startswith("https://www.tiktok.com/@username"):
            messagebox.showwarning("Invalid URL", "Please enter a valid TikTok URL.")
            return
        
        # Disable buttons during fetch
        self.fetch_button.config(state=tk.DISABLED, text="Fetching...")
        self.analyze_button.config(state=tk.DISABLED)
        
        # Run fetch in background thread
        thread = threading.Thread(
            target=self.run_fetch_transcript,
            args=(url,),
            daemon=True
        )
        thread.start()
    
    def run_fetch_transcript(self, url):
        """Fetch transcript in background thread"""
        try:
            transcript = self.extract_tiktok_transcript(url)
            self.root.after(0, lambda: self.on_transcript_fetched(transcript))
        except Exception as e:
            import traceback
            error_msg = str(e)
            print("Fetch error:", traceback.format_exc())
            self.root.after(0, lambda: self.on_fetch_error(error_msg))
    
    def extract_tiktok_transcript(self, url):
        """Extract transcript from TikTok video"""
        try:
            import yt_dlp
            import speech_recognition as sr
            import tempfile
            import os
            
            # Download audio from TikTok
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(temp_dir, 'video.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                }
                
                print(f"Downloading audio from: {url}")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Find the audio file
                audio_files = [f for f in os.listdir(temp_dir) if f.endswith('.wav')]
                if not audio_files:
                    raise Exception("Could not download audio from TikTok URL")
                
                audio_path = os.path.join(temp_dir, audio_files[0])
                
                # Transcribe audio
                print("Transcribing audio...")
                recognizer = sr.Recognizer()
                
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    
                    try:
                        transcript = recognizer.recognize_google(audio_data)
                        return transcript
                    except sr.UnknownValueError:
                        raise Exception("Could not understand audio - speech may be unclear or in another language")
                    except sr.RequestError as e:
                        raise Exception(f"Could not request results from speech recognition service: {e}")
        
        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else "unknown"
            raise Exception(
                f"Missing required library: {missing_lib}\n\n"
                f"To enable TikTok transcription, install:\n"
                f"pip install yt-dlp SpeechRecognition pydub\n\n"
                f"You also need ffmpeg installed on your system."
            )
        except Exception as e:
            raise Exception(f"Failed to extract transcript: {str(e)}")
    
    def on_transcript_fetched(self, transcript):
        """Called when transcript is successfully fetched"""
        self.fetch_button.config(state=tk.NORMAL, text="Fetch Transcript")
        self.analyze_button.config(state=tk.NORMAL)
        
        # Insert transcript
        self.transcript_text.delete("1.0", tk.END)
        self.transcript_text.insert("1.0", transcript)
        self.update_char_count()
        
        messagebox.showinfo(
            "Transcript Fetched",
            f"Successfully extracted transcript!\n\n"
            f"Length: {len(transcript)} characters\n\n"
            f"You can now analyze it or edit it first."
        )
    
    def on_fetch_error(self, error_msg):
        """Called when transcript fetch fails"""
        self.fetch_button.config(state=tk.NORMAL, text="Fetch Transcript")
        self.analyze_button.config(state=tk.NORMAL)
        
        messagebox.showerror(
            "Fetch Error",
            f"Failed to fetch transcript:\n\n{error_msg}\n\n"
            f"You can manually paste the transcript instead."
        )
    
    def load_models_async(self):
        """Load models in a background thread"""
        thread = threading.Thread(target=self.load_models, daemon=True)
        thread.start()
    
    def load_models(self):
        """Load all trained models (RF + optional GNN)"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # -------- 1. Load Random Forest model --------
            hybrid_rf_path = MODELS_DIR / "rf_hybrid_gnn.pkl"
            rf_path = MODELS_DIR / "rf_final.pkl"

            if hybrid_rf_path.exists():
                self.rf_model = joblib.load(hybrid_rf_path)
                print("✓ Loaded Hybrid RF+GNN model")
                self.model_type = "hybrid"
            elif rf_path.exists():
                self.rf_model = joblib.load(rf_path)
                print("✓ Loaded TF-IDF only RF model")
                self.model_type = "tfidf_only"
            else:
                raise FileNotFoundError("No RF model found in MODELS_DIR")

            # -------- 2. Load GNN + graph only if hybrid model is used --------
            if self.model_type == "hybrid":
                # (a) scalers for TF-IDF + GNN probability features
                scaler_path = MODELS_DIR / "hybrid_scalers.pkl"
                if scaler_path.exists():
                    self.scalers = joblib.load(scaler_path)
                    print("✓ Loaded scalers")

                # (b) graph structure (x_graph, edge_index, vocab_index)
                graph_path = MODELS_DIR / "graph_structure.pkl"
                model_paths = sorted(Path(MODELS_DIR).glob("gnn_fold*_best.pth"))

                if graph_path.exists() and model_paths:
                    graph_data = joblib.load(graph_path)
                    self.vocab_index = graph_data["vocab_index"]
                    self.x_graph = graph_data["x_graph"].to(self.device)
                    self.edge_index = graph_data["edge_index"].to(self.device)

                    input_dim = self.x_graph.shape[1]
                    print(f"✓ Loaded graph structure: {self.x_graph.shape[0]} nodes, feature dim = {input_dim}")

                    # (c) load each GNN fold, but SKIP incompatible tensors
                    for path in model_paths:
                        model = GNNClassifier(input_dim=input_dim, hidden_dim=64, dropout=0.5).to(self.device)
                        raw_state = torch.load(path, map_location=self.device)
                        model_state = model.state_dict()

                        # keep only parameters whose shapes match
                        filtered = {
                            k: v
                            for k, v in raw_state.items()
                            if k in model_state and v.size() == model_state[k].size()
                        }
                        mismatched = len(raw_state) - len(filtered)
                        if mismatched > 0:
                            print(f"⚠️ Skipping {mismatched} incompatible tensors for {path.name}")

                        model_state.update(filtered)
                        model.load_state_dict(model_state, strict=False)
                        model.eval()
                        self.gnn_models.append(model)

                    self.use_gnn = True
                    print(f"✓ Loaded {len(self.gnn_models)} GNN fold models")
                else:
                    print("⚠️ graph_structure.pkl or GNN checkpoints not found – using RF only")
                    self.use_gnn = False

            # -------- 4. Done --------
            self.models_loaded = True
            self.root.after(0, self.on_models_loaded)

        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda e=e: self.on_models_error(str(e)))

    def on_models_loaded(self):
        """Called when models load successfully"""
        self.status_label.config(text="● Models loaded - Ready", fg="#10b981")
        self.analyze_button.config(state=tk.NORMAL)
        self.fetch_button.config(state=tk.NORMAL)
        messagebox.showinfo("Ready", "Models loaded successfully!\nYou can now analyze transcripts.")
    
    def on_models_error(self, error_msg):
        """Called when model loading fails"""
        self.status_label.config(text="● Error loading models", fg="#ef4444")
        messagebox.showerror("Model Loading Error", f"Failed to load models:\n{error_msg}")
    
    def analyze_transcript(self):
        """Start analyzing the transcript."""
        if not self.models_loaded:
            messagebox.showwarning("Not Ready", "Models are still loading. Please wait.")
            return

        transcript = self.transcript_text.get("1.0", tk.END).strip()

        if not transcript:
            messagebox.showwarning("Missing Input", "Please enter a transcript or URL.")
            return

        # Disable UI components to prevent multiple submissions
        self.analyze_button.config(state=tk.DISABLED, text="Analyzing...")
        self.results_container.grid_remove()

        # Run analysis in a separate thread
        thread = threading.Thread(target=self.run_analysis, args=(transcript,), daemon=True)
        thread.start()

    
    def run_analysis(self, transcript):
        """Run the analysis (prediction) in a separate thread"""
        try:
            # Preprocess and run predictiona
            tokens = preprocess_pipeline(transcript)  # Tokenize and clean the text
            result = self.predict_transcript(tokens)  # Pass preprocessed tokens to the prediction method
            self.root.after(0, lambda: self.display_results(result))  # Update UI after inference
        except Exception as e:
            self.root.after(0, lambda: self.on_analysis_error(str(e)))

    def predict_transcript(self, tokens):
        """Predict if transcript is fake or real"""
        try:
            if not self.vectorizer or not self.rf_model:
                raise ValueError("Models not loaded properly")
            
            # Rebuild the text from tokens for TF-IDF vectorization
            text = " ".join(tokens)  # Reconstruct the text from tokens
            X_tfidf = self.vectorizer.transform([text])

            # Get GNN features if hybrid model is used
            if self.use_gnn and self.gnn_models and self.vocab_index and self.x_graph is not None:
                all_probs = []
                for model in self.gnn_models:
                    try:
                        with torch.no_grad():
                            probs = extract_gnn_probabilities(
                                model, self.x_graph, self.edge_index,
                                [tokens], self.vocab_index,
                                tfidf_matrix=X_tfidf,
                                device=self.device
                            )
                        all_probs.append(probs)
                    except:
                        all_probs.append(np.array([[0.5]]))  # Default to 50% if GNN fails

                all_probs = np.stack(all_probs, axis=0)  # Stack the outputs of all GNN models
                weights = np.ones(len(self.gnn_models)) / len(self.gnn_models)  # Equal weight for all models
                gnn_prob_weighted = np.mean(all_probs, axis=0)  # Combine GNN probabilities
                
                # Scale the features if scalers are available
                if self.scalers:
                    X_scaled = self.scalers['tfidf_scaler'].transform(X_tfidf)
                    gnn_scaled = self.scalers['emb_scaler'].transform(gnn_prob_weighted.reshape(-1, 1))
                    gnn_scaled *= self.scalers['gnn_scale_factor']
                # else:
                #     # If no scalers, apply default scaling
                #     tfidf_scaler = MaxAbsScaler()
                #     X_scaled = tfidf_scaler.fit_transform(X_tfidf)
                #     emb_scaler = StandardScaler()
                #     gnn_block = gnn_prob_weighted
                #     if gnn_block.ndim == 1:
                #         gnn_block = gnn_block.reshape(-1, 1)
                #     gnn_scaled = emb_scaler.fit_transform(gnn_block)
                #     gnn_scaled *= 0.90

            # 1. TF-IDF scaling (use saved scaler; DO NOT FIT AGAIN)
            if self.scalers:
                tfidf_scaled = self.scalers["tfidf_scaler"].transform(X_tfidf)
            else:
                tfidf_scaled = MaxAbsScaler().fit_transform(X_tfidf)

            # 2. GNN scaling (use saved emb_scaler; DO NOT FIT AGAIN)
            if self.scalers:
                gnn_scaled = self.scalers["emb_scaler"].transform(gnn_prob_weighted.reshape(-1, 1))
                gnn_scaled *= self.scalers["gnn_scale_factor"]
            else:
                gnn_scaled = StandardScaler().fit_transform(gnn_prob_weighted.reshape(-1, 1))
                gnn_scaled *= 0.90

            # 3. Convert to sparse & combine EXACTLY like hybrid_rf_gnn.py
            gnn_sparse = sp.csr_matrix(gnn_scaled)
            X_combined = sp.hstack([tfidf_scaled, gnn_sparse], format="csr")  
            # else:
            #     # If no GNN model, only use TF-IDF
            #     X_combined = X_tfidf
            
            # Predict using Random Forest
            prediction = self.rf_model.predict(X_combined)[0]
            probability = self.rf_model.predict_proba(X_combined)[0]
            
            fake_prob = probability[1]  # Probability of being fake news
            
            return {
                "prediction": "fake" if prediction == 1 else "real",
                "probability": float(fake_prob),
                "token_count": len(tokens)
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise


    
    def display_results(self, result):
        """Display results after prediction"""
        self.analyze_button.config(state=tk.NORMAL, text="Analyze Transcript")
        self.results_container.grid()
        
        if result["prediction"] == "fake":
            self.result_label.config(text="⚠️ FAKE NEWS DETECTED", fg="#ef4444")
        else:
            self.result_label.config(text="✓ LIKELY REAL NEWS", fg="#10b981")
        
        prob_pct = result["probability"] * 100
        self.prob_label.config(text=f"{prob_pct:.2f}%")
        self.prob_bar['value'] = prob_pct

        # Change the color of the progress bar based on probability
        style = ttk.Style()
        if prob_pct >= 70:
            style.configure("TProgressbar", background="#ef4444")  # Red for fake
        elif prob_pct >= 50:
            style.configure("TProgressbar", background="#f59e0b")  # Orange for uncertain
        else:
            style.configure("TProgressbar", background="#10b981")  # Green for real

        self.root.update_idletasks()

    
    def on_analysis_error(self, error_msg):
        """Called when an error occurs during analysis"""
        messagebox.showerror("Analysis Error", f"Failed to analyze transcript:\n{error_msg}")
        self.analyze_button.config(state=tk.NORMAL, text="Analyze Transcript")
        self.results_container.grid_remove()



def main():
    root = tk.Tk()
    app = FakeTTAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

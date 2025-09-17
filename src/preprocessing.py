"""
Handles text preprocessing for FakeTT transcripts:
- Lowercasing
- Removing puncation and special characters
- Tokenization
- Stopword removal
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Download stopwords & tokenizer (run once)

# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9#\s]", "", text)  # keep letters, numbers, hashtags, spaces
    text = re.sub(r"\s+", " ", text).strip()     # remove extra spaces
    return text

def tokenize(text: str) -> list:
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: list, extra_stopwords: list = None) -> list:
    stop_words = set(stopwords.words("english"))
    if extra_stopwords:
        stop_words.update(extra_stopwords)
    return [word for word in tokens if word not in stop_words]

def preprocess_pipeline(text: str) -> list:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    return tokens

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]
    if "transcript" not in df.columns:
        raise ValueError("Transcript column required")
    
    initial_len = len(df) #rows before removing empty transcripts

    df = df.dropna(subset=["transcript"])
    df = df[df["transcript"].str.strip() != ""]
    
    cleaned_len = len(df) #rows after removing empty transcripts

    dropped = initial_len - cleaned_len

    print(f"Cleaned Dataset. {dropped} rows were removed due to having empty transcripts")    
    return df

def preprocess_dataset(csv_path: str) -> pd.DataFrame:
     df = load_dataset(csv_path)
     df["tokens"] = df["transcript"].apply(preprocess_pipeline)
     return df

def save_preprocessed_dataset(df: pd.DataFrame, output_path:str):

    df.to_csv(output_path, index=False)

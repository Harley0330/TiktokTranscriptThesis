"""
preprocessing_plots.py (tokens-aware)
Builds preprocessing & feature-extraction visuals using the existing `tokens`
column and removes stopwords before plotting.
"""

import ast
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import text as sk_text
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "data_cleaned_formatted.csv"
PLOTS_DIR = BASE_DIR / "plots" / "preprocessing_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# === Load ===
df = pd.read_csv(DATA_PATH)
for col in ("transcript", "tokens"):
    if col not in df.columns:
        raise ValueError(f"Dataset must contain '{col}' column.")
print(f"Loaded {len(df)} rows")

# ---- Parse tokens column if stored as string ----
def parse_tokens(x):
    if isinstance(x, list):
        return [str(t) for t in x]
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:
            # fall back: split on spaces
            return x.split()
    return []

df["tokens_list"] = df["tokens"].apply(parse_tokens)

# ---- Stopwords (generic + domain-specific) ----
generic_sw = set(sk_text.ENGLISH_STOP_WORDS)
domain_sw = {
    "video","shows","tiktok","people","say","saying","like","just","really",
    "look","looks","thing","things","man","woman","guy","girl","im","ive","you",
    "get","got","going","one","two","three","make","made","see","seen","today",
    "news","clip","watch","live"
}
STOPWORDS = generic_sw | domain_sw

def remove_stopwords(tokens):
    return [t for t in tokens if t and t.lower() not in STOPWORDS and t.isalpha()]

df["tokens_filt"] = df["tokens_list"].apply(remove_stopwords)

# =============== 1) Most frequent tokens (from filtered tokens) ===============
flat = [t.lower() for row in df["tokens_filt"] for t in row]
word_counts = Counter(flat)
top20 = word_counts.most_common(20)

plt.figure(figsize=(8,5))
sns.barplot(x=[c for _, c in top20], y=[w for w, _ in top20], palette="crest")
plt.title("Top 20 Tokens (after stopword removal)")
plt.xlabel("Frequency"); plt.ylabel("")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "token_frequency.png", dpi=300); plt.close()

# =============== 2) Word cloud (from filtered tokens) =========================
wc_text = " ".join(flat)
wc = WordCloud(width=1200, height=600, background_color="white").generate(wc_text)
plt.figure(figsize=(10,5)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
plt.title("Word Cloud (after stopword removal)")
plt.tight_layout(); plt.savefig(PLOTS_DIR / "wordcloud.png", dpi=300); plt.close()

# =============== 3) TF-IDF (built from filtered tokens) =======================
docs_for_tfidf = [" ".join(row) for row in df["tokens_filt"]]
vectorizer = TfidfVectorizer(max_features=1000, lowercase=False, stop_words=None)
X = vectorizer.fit_transform(docs_for_tfidf)

# Top 20 global TF-IDF terms
mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
top_idx = mean_tfidf.argsort()[-20:][::-1]
top_terms = np.array(vectorizer.get_feature_names_out())[top_idx]
plt.figure(figsize=(8,5))
sns.barplot(x=mean_tfidf[top_idx], y=top_terms, palette="mako")
plt.title("Top 20 TF-IDF Terms (after stopword removal)")
plt.xlabel("Mean TF-IDF weight"); plt.ylabel("")
plt.tight_layout(); plt.savefig(PLOTS_DIR / "tfidf_top_terms.png", dpi=300); plt.close()

# =============== 4) Co-occurrence graph (from filtered tokens) ===============
def build_cooccurrence_graph(tokens_rows, top_k=100):
    co = Counter()
    for toks in tokens_rows:
        uniq = sorted(set(toks))
        for i in range(len(uniq)):
            for j in range(i+1, len(uniq)):
                co[(uniq[i], uniq[j])] += 1
    G = nx.Graph()
    for (w1, w2), c in co.most_common(top_k):
        G.add_edge(w1, w2, weight=c)
    return G

print("Building co-occurrence graph...")
G = build_cooccurrence_graph(df["tokens_filt"].head(1000), top_k=100)

plt.figure(figsize=(9,7))
pos = nx.spring_layout(G, k=0.35, seed=42)
nx.draw_networkx(
    G, pos,
    node_color="skyblue", node_size=500,
    edge_color="gray", width=[G[u][v]["weight"]*0.1 for u,v in G.edges()],
    font_size=8
)
plt.title("Word Co-occurrence Network (top 100 pairs, stopwords removed)")
plt.tight_layout(); plt.savefig(PLOTS_DIR / "word_cooccurrence_network.png", dpi=300); plt.close()

# Degree histogram
degrees = [d for _, d in G.degree()]
plt.figure(figsize=(6,4))
plt.hist(degrees, bins=15, color="skyblue", edgecolor="gray")
plt.title("Co-occurrence Graph Degree Distribution")
plt.xlabel("Degree"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(PLOTS_DIR / "word_graph_degree_hist.png", dpi=300); plt.close()

print(f"Done. Saved plots to: {PLOTS_DIR}")

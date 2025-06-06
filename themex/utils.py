import re
import sys
import os
import csv
import json
import spacy
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Set, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import torch
import time
import tracemalloc
import functools
from jsonfinder import jsonfinder

from .paths import PROMPT_DIR
from .logger import get_logger

logger = get_logger()

# TODO: need further tidy up, some functions should be stored in seperate file

# ---------------- WC ----------------

# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk import word_tokenize, pos_tag
# from nltk.corpus import wordnet

# nltk_resources = {
#     'punkt': 'tokenizers/punkt',
#     'wordnet': 'corpora/wordnet',
#     'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
#     'omw-1.4': 'corpora/omw-1.4'
# }

# print("🔍 Checking NLTK resources...")
# for name, path in nltk_resources.items():
#     try:
#         nltk.data.find(path)
#         print(f"✅ NLTK resource '{name}' is available.")
#     except LookupError:
#         print(f"⬇️  Downloading missing NLTK resource: {name}")
#         nltk.download(name)

# # Check if spaCy and en_core_web_sm are installed
# print("\n🔍 Checking spaCy model 'en_core_web_sm'...")
# try:
#     spacy.load("en_core_web_sm")
#     print("✅ spaCy model 'en_core_web_sm' is available.")
# except OSError:
#     print("⬇️  Downloading spaCy model 'en_core_web_sm'...")
#     subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
#     print("✅ spaCy model 'en_core_web_sm' installed successfully.")

# nlp = spacy.load("en_core_web_sm")

def suggest_stopwords_from_df(
    df: pd.DataFrame,
    text_columns: List[str],
    min_df: int = 5,
    top_n: int = 50,
    whitelist: Optional[Set[str]] = None,
    idf_cutoff: float = 1.8
) -> pd.DataFrame:
    """
    Automatically suggest stopwords from text columns using spaCy-based
    lemmatization and POS tagging, along with TF-IDF statistics.

    Parameters:
        df (pd.DataFrame): The input dataframe containing text data.
        text_columns (List[str]): List of column names in df that contain text.
        min_df (int, default=5): Minimum document frequency to include a word in IDF calculation.
        top_n (int, default=50): Number of top frequent words to return in the output.
        whitelist (Optional[Set[str]]): Words to exclude from being marked as stopword candidates.
        idf_cutoff (float, default=1.8): Words with IDF below this threshold are considered common.

    Returns:
        pd.DataFrame: A DataFrame with columns: 'word', 'tf', 'idf', 'pos', 'suggest_stopword'.
                      This shows the most common words and whether they are suggested as stopwords.
    """
    if whitelist is None:
        whitelist = set()

    # Combine all specified columns into one text block per row
    docs = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1).tolist()

    all_lemmas = []
    lemmatized_docs = []

    for doc in docs:
        cleaned = preprocess_text(doc)
        lemmatized_docs.append(cleaned)
        all_lemmas.extend(cleaned.split())

    # Word frequency
    tf_counter = Counter(all_lemmas)

    # TF-IDF model
    vectorizer = TfidfVectorizer(min_df=min_df)
    vectorizer.fit(lemmatized_docs)
    idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    # POS tagging for top frequent words
    most_common_words = [w for w, _ in tf_counter.most_common(500)]
    pos_map = {}
    for word in most_common_words:
        doc = nlp(word)
        if len(doc) > 0:
            pos_map[word] = doc[0].pos_

    # Compile result table
    rows = []
    for word in most_common_words:
        tf = tf_counter[word]
        idf = idf_scores.get(word, np.nan)
        pos = pos_map.get(word, '')

        is_candidate = (
            idf < idf_cutoff and
            pos in {'VERB', 'ADJ', 'ADV'} and
            word not in whitelist
        )

        rows.append({
            'word': word,
            'tf': tf,
            'idf': round(idf, 3) if not np.isnan(idf) else 'N/A',
            'pos': pos,
            'suggest_stopword': is_candidate
        })

    return pd.DataFrame(rows).sort_values(by='tf', ascending=False).head(top_n)

# Preprocess text: lowercase + tokenization + lemmatization
def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing special whitespace, lowercasing, 
    lemmatizing (with POS), and removing non-alphabetic tokens.

    Args:
        text (str): Raw text input.

    Returns:
        str: Cleaned and lemmatized text string.
    """
    cleaned_text = re.sub(r'[\u200b\u200c\u200d\u2060\u00a0]', '', text.lower())
    doc = nlp(cleaned_text)

    lemmatized_words = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and token.lemma_ not in STOPWORDS and token.lemma_.lower() not in string.ascii_lowercase
    ]
    
    return ' '.join(lemmatized_words)
  
# Generate wordcloud matrix
def generate_wordcloud_matrix(
    df: pd.DataFrame,
    comment_columns: List[str],
    groupby_col: str,
    col_titles: List[str],
    custom_stopwords: Set[str] = None,
):
    """
    Generate a matrix of word clouds for multiple comment types across time periods or groups.

    Args:
        df (pd.DataFrame): Input DataFrame containing text data.
        comment_columns (List[str]): List of comment columns to visualize.
        groupby_col (str): Column to group by (e.g. Year_Quarter).
        col_titles (List[str]): Ordered group values (e.g. sorted list of quarters).
        custom_stopwords (Optional[Set[str]]): Additional global stopwords to exclude.
    """
    # Global stopwords: base + optional custom
    base_stopwords = set(STOPWORDS)
    base_stopwords.update(string.ascii_lowercase)
    if custom_stopwords:
        base_stopwords.update({w.lower() for w in custom_stopwords})

    # Per-column raw stopwords (to be lemmatized)
    raw_stopwords_dict = {
        'Rights Comments': {'right'},
        # 'Assets Comments': {'assets', 'ownership'},
        'Risk Comments': {'risk'},
        'Legislation Comments': {'care', 'act', 'legislation', 'right'},
        'Strength Comments': {'strength'},
    }

    # Lemmatize per-column stopwords using spaCy
    lemmatized_stopwords_dict = {
        col: {
            token.lemma_.lower()
            for word in stop_set
            for token in nlp(word.lower())
            if token.is_alpha
        }
        for col, stop_set in raw_stopwords_dict.items()
    }

    n_rows = len(comment_columns)
    n_cols = len(col_titles)

    _, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(3.5 * n_cols, 3 * n_rows),
        constrained_layout=True
    )

    for row_idx, comment_col in enumerate(comment_columns):
        # Group and lemmatize text per group
        grouped = df.groupby(groupby_col)[comment_col].apply(
            lambda x: preprocess_text(' '.join(x.dropna().astype(str)))
        )

        # Combine stopwords for this comment column
        this_stopwords = base_stopwords.union(lemmatized_stopwords_dict.get(comment_col, set()))

        for col_idx, group_val in enumerate(col_titles):
            ax = axs[row_idx][col_idx] if n_rows > 1 else axs[col_idx]
            text = grouped.get(group_val, '')

            if text.strip():
                wc = WordCloud(
                    width=400,
                    height=300,
                    margin=0,
                    background_color='white',
                    stopwords=this_stopwords,
                    colormap='viridis'
                ).generate(text)
                ax.imshow(wc, interpolation='bilinear')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)

            ax.axis('off')
            if row_idx == 0:
                ax.set_title(group_val, fontsize=11, pad=10)

    # Add left-side row labels (comment type names)
    for row_idx, comment_col in enumerate(comment_columns):
        ax = axs[row_idx][0] if n_rows > 1 else axs[0]
        ax.text(
            -0.2, 0.5, comment_col,
            transform=ax.transAxes,
            ha='right', va='center',
            fontsize=11, fontweight='bold'
        )

    plt.show()

# # Function to download model if missing
# def download_model():
#     os.makedirs(MODEL_FOLDER, exist_ok=True)
#     if not os.path.exists(MODEL_PATH):
#         print("Downloading LLaMA GGUF model...")
#         with requests.get(MODEL_URL, stream=True) as r:
#             r.raise_for_status()
#             total = int(r.headers.get('content-length', 0))
#             with open(MODEL_PATH, 'wb') as f, tqdm(
#                 desc=MODEL_NAME,
#                 total=total,
#                 unit='iB',
#                 unit_scale=True,
#                 unit_divisor=1024,
#             ) as bar:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)
#                     bar.update(len(chunk))
#         print("Model download completed.")


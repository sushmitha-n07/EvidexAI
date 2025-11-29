# nlp/rag_retriever.py
import os
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_firs(query_text: str, corpus: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Find FIRs in the corpus most similar to the query text using TF-IDF + cosine similarity.

    Args:
        query_text (str): The input FIR text.
        corpus (List[str]): List of FIR texts.
        top_k (int): Number of top similar FIRs to return.

    Returns:
        List[Tuple[str, float]]: List of (FIR text, similarity score).
    """
    if not corpus:
        return [("No similar cases available", 0.0)]

    # Preprocess
    query_text = query_text.strip().lower()
    corpus_clean = [doc.strip().lower() for doc in corpus]

    # Vectorize
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query_text] + corpus_clean)

    # Compute similarity
    scores = cosine_similarity(vectors[0:1], vectors[1:])[0]

    # Rank results
    ranked = sorted(zip(corpus, scores), key=lambda x: -x[1])

    return ranked[:top_k]

def load_fir_corpus(path: str) -> List[str]:
    """
    Load FIR corpus from text files in a directory.
    Each .txt file is read and added to the corpus list.
    """
    corpus = []
    if os.path.exists(path):
        for fname in os.listdir(path):
            if fname.endswith(".txt"):
                try:
                    with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                        corpus.append(f.read())
                except Exception:
                    continue
    return corpus
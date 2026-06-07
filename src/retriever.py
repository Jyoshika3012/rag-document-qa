import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_chunks(query: str, index_path: str = "data/faiss_index", k: int = 4):
    with open(f"{index_path}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    texts = [chunk.page_content for chunk in chunks]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_k = np.argsort(scores)[::-1][:k]

    return [chunks[i] for i in top_k]
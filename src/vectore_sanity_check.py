from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from vectore_store import build_faiss_index

def search(query, index, model, df, k=5):

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, k)

    results = df.iloc[indices[0]]

    return results


if __name__ == "__main__":

    df = pd.read_csv("data/processed/cleaned_articles.csv")
    embeddings = np.load("data/processed/article_embeddings.npy")

    index = build_faiss_index(embeddings)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "latest news about artificial intelligence"

    results = search(query, index, model, df)

    print(results[["headline", "category"]])
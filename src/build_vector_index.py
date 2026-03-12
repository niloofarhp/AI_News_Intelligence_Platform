import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer


# ----------------------------
# Paths
# ----------------------------

RAW_DATA_PATH = "data/raw/News_Category_Dataset_v3.json"
PROCESSED_DATA_PATH = "data/processed/cleaned_articles.csv"
EMBEDDINGS_PATH = "data/processed/article_embeddings.npy"
FAISS_INDEX_PATH = "data/processed/faiss_index.bin"


# ----------------------------
# Load and preprocess dataset
# ----------------------------

def load_dataset():

    print("Loading dataset...")

    data = []

    with open(RAW_DATA_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    df["full_text"] = df["headline"] + ". " + df["short_description"]

    df = df[["headline", "short_description", "category", "date", "full_text"]]

    return df


# ----------------------------
# Generate embeddings
# ----------------------------

def generate_embeddings(df):

    print("Generating embeddings...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["full_text"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True
    )

    return embeddings


# ----------------------------
# Build FAISS index
# ----------------------------

def build_faiss_index(embeddings):

    print("Building FAISS index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# ----------------------------
# Main pipeline
# ----------------------------

def main():

    os.makedirs("data/processed", exist_ok=True)

    df = load_dataset()

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    embeddings = generate_embeddings(df)

    np.save(EMBEDDINGS_PATH, embeddings)

    index = build_faiss_index(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    print("Vector index successfully created!")


# ----------------------------

if __name__ == "__main__":
    main()
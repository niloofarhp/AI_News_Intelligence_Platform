import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_embeddings(texts):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64
    )

    return embeddings


if __name__ == "__main__":

    input_path = "data/processed/cleaned_articles.csv"
    output_path = "data/processed/article_embeddings.npy"

    df = pd.read_csv(input_path)

    texts = df["full_text"].tolist()

    embeddings = generate_embeddings(texts)

    np.save(output_path, embeddings)

    print("Embeddings saved.")
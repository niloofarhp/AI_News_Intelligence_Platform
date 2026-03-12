import numpy as np
import faiss


def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


if __name__ == "__main__":

    embeddings_path = "data/processed/article_embeddings.npy"
    index_path = "data/processed/faiss_index.bin"

    embeddings = np.load(embeddings_path)

    index = build_faiss_index(embeddings)

    faiss.write_index(index, index_path)

    print("FAISS index saved.")
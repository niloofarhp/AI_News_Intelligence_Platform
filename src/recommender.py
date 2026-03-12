import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class NewsRecommender:

    def __init__(self, index_path, data_path):

        self.index = faiss.read_index(index_path)

        self.df = pd.read_csv(data_path)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")


    def recommend_from_text(self, article_text, k=5):

        embedding = self.model.encode([article_text])

        distances, indices = self.index.search(embedding, k + 1)

        recommendations = self.df.iloc[indices[0]]

        return recommendations.iloc[1:]


    def recommend_from_id(self, article_id, k=5):

        article = self.df.iloc[article_id]

        text = article["full_text"]

        return self.recommend_from_text(text, k)

    def recommend_from_queries(self, queries, k=5):

        embeddings = self.model.encode(queries)

        user_embedding = embeddings.mean(axis=0).reshape(1, -1)

        distances, indices = self.index.search(user_embedding, k)

        return self.df.iloc[indices[0]]
    
if __name__ == "__main__":

    recommender = NewsRecommender(
        index_path="data/processed/faiss_index.bin",
        data_path="data/processed/cleaned_articles.csv"
    )

    article_id = 0

    recommendations = recommender.recommend_from_id(article_id)

    print("Recommended Articles:")
    print(recommendations[["headline", "category"]])    
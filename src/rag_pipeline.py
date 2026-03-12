import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os


state = load_dotenv()

client = OpenAI()


class RAGPipeline:

    def __init__(self, index_path, data_path):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index = faiss.read_index(index_path)

        self.df = pd.read_csv(data_path)

    def retrieve(self, query, k=5):

        query_embedding = self.model.encode([query])

        distances, indices = self.index.search(query_embedding, k)

        results = self.df.iloc[indices[0]]

        return results

    def build_context(self, retrieved_articles):

        context = ""

        for _, row in retrieved_articles.iterrows():
            context += f"Headline: {row['headline']}\n"
            context += f"Description: {row['short_description']}\n\n"

        return context

    def generate_answer(self, query, context):

        prompt = f"""
                    You are an AI news assistant.

                    Use only the provided news context to answer the question.

                    Context:
                    {context}

                    Question:
                    {query}

                    Answer:
                    """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def ask(self, query):

        retrieved = self.retrieve(query)

        context = self.build_context(retrieved)

        answer = self.generate_answer(query, context)

        return answer, retrieved
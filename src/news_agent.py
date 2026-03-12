from openai import OpenAI
from dotenv import load_dotenv
import os

from src.rag_pipeline import RAGPipeline
from src.recommender import NewsRecommender
from src.summarizer import summarize_article

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class NewsAgent:

    def __init__(self):

        self.rag = RAGPipeline(
            "data/processed/faiss_index.bin",
            "data/processed/cleaned_articles.csv"
        )

        self.recommender = NewsRecommender(
            "data/processed/faiss_index.bin",
            "data/processed/cleaned_articles.csv"
        )
        self.conversation_history = []
        self.user_queries = []

    def decide_action(self, query):

        prompt = f"""
You are an AI assistant that decides which tool to use.

Tools available:
1. search_news
2. answer_question
3. summarize_news
4. recommend_articles
5. recommend_from_history

User query:
{query}

Return only the tool name.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()


    def run(self, query):

        action = self.decide_action(query)

        if action == "search_news":
            print("Action: Search News")
            self.user_queries.append(query)
            return self.rag.retrieve(query)

        elif action == "answer_question":
            print("Action: Answer Question")
            self.user_queries.append(query)
            answer, _ = self.rag.ask(query)
            return answer

        elif action == "recommend_articles":
            print("Action: Recommend Articles")
            return self.recommender.recommend_from_queries([query])

        elif action == "summarize_news":
            print("Action: Summarize News")

            results = self.rag.retrieve(query)

            text = results.iloc[0]["full_text"]

            return summarize_article(text)
        elif action == "recommend_from_history":
            print("Action: Recommend Based on History")

            if not self.user_queries:
                return "No user queries in history to base recommendations on."

            return self.recommender.recommend_from_queries(self.user_queries)

        else:
            return "I couldn't determine the correct action."
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

client = OpenAI()


def summarize_article(article_text):

    prompt = f"""
                Summarize the following news article in 3 concise bullet points.

                Article:
                {article_text}
                """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":

    df = pd.read_csv("data/processed/cleaned_articles.csv")

    article = df.iloc[0]["full_text"]

    summary = summarize_article(article)

    print("Article Summary:")
    print(summary)
import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.recommender import NewsRecommender
from src.summarizer import summarize_article

import pandas as pd


if "search_history" not in st.session_state:
    st.session_state.search_history = []
    
st.set_page_config(
    page_title="NewsMind AI",
    page_icon="📰",
    layout="wide"
)

st.title("📰 NewsMind AI")
st.subheader("AI-Powered News Intelligence Platform")

st.sidebar.subheader("Your Interests")

st.sidebar.write(st.session_state.search_history)

@st.cache_resource
def load_systems():

    rag = RAGPipeline(
        index_path="data/processed/faiss_index.bin",
        data_path="data/processed/cleaned_articles.csv"
    )

    recommender = NewsRecommender(
        index_path="data/processed/faiss_index.bin",
        data_path="data/processed/cleaned_articles.csv"
    )

    df = pd.read_csv("data/processed/cleaned_articles.csv")

    return rag, recommender, df


rag, recommender, df = load_systems()

st.sidebar.title("Features")


option = st.sidebar.selectbox(
    "Choose a tool",
    [
        "Ask News (RAG)",
        "Semantic Search",
        "Summarize Article",
        "Recommend Articles",
        "Recommend Based on My Interests"
    ]
)

if option == "Ask News (RAG)":

    st.header("Ask Questions About News")

    question = st.text_input("Enter your question")

    if st.button("Ask"):

        st.session_state.search_history.append(question)
        
        answer, articles = rag.ask(question)

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Retrieved Articles")

        st.dataframe(articles[["headline", "category"]])
        
elif option == "Semantic Search":

    st.header("Semantic News Search")

    query = st.text_input("Search News")

    if st.button("Search"):
        
        st.session_state.search_history.append(query)
        
        results = rag.retrieve(query)

        st.dataframe(results[["headline", "category"]])  
              
elif option == "Summarize Article":

    st.header("Article Summarization")

    article_id = st.number_input("Article Index", min_value=0, max_value=len(df)-1)

    if st.button("Summarize"):

        article = df.iloc[article_id]["full_text"]

        summary = summarize_article(article)

        st.subheader("Summary")

        st.write(summary)  
     
elif option == "Recommend Articles":

    st.header("Recommended Articles")

    article_id = st.number_input("Article Index", min_value=0, max_value=len(df)-1)

    if st.button("Recommend"):

        recs = recommender.recommend_from_id(article_id)

        st.dataframe(recs[["headline", "category"]])                    

elif option == "Recommend Based on My Interests":

    st.header("Personalized News Recommendations")

    history = st.session_state.search_history

    if len(history) == 0:

        st.write("Search for some news first to build your interests.")

    else:

        st.write("Your recent interests:")

        st.write(history)

        if st.button("Generate Recommendations"):

            recs = recommender.recommend_from_queries(history)

            st.subheader("Recommended Articles")

            st.dataframe(recs[["headline", "category"]])
            
            
if st.sidebar.button("Clear History"):
    st.session_state.search_history = []                    
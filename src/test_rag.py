from rag_pipeline import RAGPipeline

rag = RAGPipeline(
    index_path="data/processed/faiss_index.bin",
    data_path="data/processed/cleaned_articles.csv"
)

query = "What are the latest developments in artificial intelligence?"

answer, articles = rag.ask(query)

print("Answer:\n", answer)

print("\nRetrieved Articles:\n")

print(articles[["headline", "category"]])
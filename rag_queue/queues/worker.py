from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def process_query(query: str):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    vector_db = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="learning_rag",
        embedding=embedding_model
    )
    search_result = vector_db.similarity_search(query=query)
    seen = set()
    search_result = [r for r in search_result if not (r.page_content in seen or seen.add(r.page_content))]
    print(f"\n--- Search Results ({len(search_result)} chunks) ---")
    for i, r in enumerate(search_result):
        print(f"[{i+1}] Page {r.metadata.get('page_label')} | {r.page_content[:200]}")
    print("---\n")

    context = "\n\n\n".join([
        f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page')}\nPage Label: {result.metadata.get('page_label')}\nSource: {result.metadata.get('source')}"
        for result in search_result
    ])

    SYSTEM_PROMPT = f"""
You are a helpful AI assistant who answers user query based on the avaliable context
retrieved from a PDF file along with page_contents and page number.

You should only answer the user based on the following context and navigate the user to open the right page number to know more.
Context:
{context}

"""

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content

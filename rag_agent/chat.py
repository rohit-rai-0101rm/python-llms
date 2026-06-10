from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)

user_query=input("Ask something")

search_result=vector_db.similarity_search(query=user_query)


context = "\n\n\n".join([
    f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page')}\nPage Label: {result.metadata.get('page_label')}\nSource: {result.metadata.get('source')}"
    for result in search_result
])

SYSTEM_PRPMPT=f"""
You are a helpful AI assistant who answers user query based on the avaliable context
retrieved froma PDF file along with page_contents and page number.

You should only ans the user based on the following context and navigate the user to oepn the right page number to know more
Context:
{context}

"""


client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content":SYSTEM_PRPMPT
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
)

print(f"{response.choices[0].message.content}")
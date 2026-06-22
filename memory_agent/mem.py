import os
os.environ["MEM0_TELEMETRY"] = "False"  # belt-and-suspenders, silence telemetry thread
from dotenv import load_dotenv
import time
from mem0 import Memory
import ollama
load_dotenv() 

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# Mem0 Configuration for a 100% Local Pipeline
config = {
    "version": "v1.1",
    "telemetry_enabled": False,  # Disables background analytics tracking
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-mpnet-base-v2"
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "ollama_base_url": "http://localhost:11434"
        }
    },

  "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URI,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768
        }
    }
}

# Initialize Mem0
mem_client = Memory.from_config(config)

print("--- 100% Local Memory Agent Initialized ---")
print("Using: Gemma-2B (Ollama Docker) & Hugging Face Embeddings")
print("Type 'exit' or 'quit' to end the session.\n")

USER_ID = "rohit"

# Interactive Chat Loop
while True:
    user_query = input("> ")

    if user_query.strip().lower() in ["exit", "quit"]:
        print("Ending session. Goodbye!")
        break

    if not user_query.strip():
        continue

    # 1. RETRIEVE ALL MEMORIES FROM THE DATABASE FOR ROBUST RECALL
    context_list = []
    try:
        # New mem0 API requires filters dict, not direct user_id kwarg
        all_memories = mem_client.get_all(filters={"user_id": USER_ID})

        # Response shape varies by version: dict with "results" key, or raw list
        if isinstance(all_memories, dict):
            records = all_memories.get("results", [])
        elif isinstance(all_memories, list):
            records = all_memories
        else:
            records = []

        for m in records:
            if isinstance(m, dict):
                text_val = m.get("memory") or m.get("data") or m.get("text")
                if text_val:
                    context_list.append(text_val)

        context = "\n".join(context_list) if context_list else "No stored memories found in the database."
        print(f"\n[System Recall - All Stored Facts]:\n{context}\n")
    except Exception as e:
        context = "No stored memories found in the database."
        print(f"[System Notice: Failed to retrieve database context: {e}]")

    # 2. GENERATE AI RESPONSE VIA OLLAMA (DOCKER)
    try:
        system_prompt = f"You are a helpful assistant. Here is everything you currently remember about the user:\n{context}"

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        ai_response = response["message"]["content"]
        print("AI Response:", ai_response)

    except Exception as e:
        ai_response = "[Failed to generate response locally]"
        print(f"\nOllama Local Response Error: {e}")
        continue

    # 3. SAVE NEW FACTS INTO MEMORY — infer=True lets the LLM extract entities for Neo4j graph
    try:
        add_result = mem_client.add(
            user_query,
            user_id=USER_ID,
            infer=True
        )
        print(f"\n[System: Memory updated successfully! Raw result: {add_result}]\n")
    except Exception as e:
        print(f"\n[System: Memory save failed: {e}]\n")

    time.sleep(0.5)
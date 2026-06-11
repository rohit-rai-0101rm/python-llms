# RAG Queue — Deep Explanation

## What is this project?

This project upgrades the basic RAG pipeline (ask a question → get an answer) into a **production-ready API** using:

- **FastAPI** — to expose HTTP endpoints
- **RQ (Redis Queue)** — to handle heavy LLM jobs in the background
- **Valkey (Redis)** — as the message broker that holds the job queue
- **Qdrant** — vector database that stores PDF embeddings
- **Gemini** — the LLM that generates the final answer

---

## The Problem with Simple RAG

In the basic `rag_agent/chat.py`, everything runs one after another:

```
User types query
    → search Qdrant (network call, ~1s)
    → call Gemini API (network call, 3-10s)
    → print answer
```

This is fine for one user in a terminal. But in a real API:

- If 100 users send a request at the same time, the server handles them one by one
- Each user waits 10+ seconds
- The server can crash under load

**Solution: Queue the heavy work and respond immediately.**

---

## Architecture — The Restaurant Analogy

```
User (customer)
    |
    | POST /chat  {"query": "what is authentication?"}
    v
server.py  <-- Waiter: takes your order, gives you a ticket number immediately
    |
    | queue.enqueue(process_query, query)
    v
rq_client.py  <-- The order rail between waiter and kitchen
    |
    | Job stored in Valkey (Redis)
    v
worker.py  <-- Chef: picks up the job, does the heavy work
    |
    | searches Qdrant + calls Gemini
    v
Result stored back in Redis

User polls GET /job-status?job_id=abc123
    → waiter checks the kitchen → "ready!" → returns the answer
```

---

## File-by-File Breakdown

### `main.py` — Entry Point

```python
from .server import app
from dotenv import load_dotenv
import uvicorn

load_dotenv()

def main():
    uvicorn.run(app, port=8000, host="0.0.0.0")

main()
```

**What it does:**
- Loads environment variables from `.env` (API keys etc.)
- Starts the FastAPI web server using `uvicorn` on port `8000`
- `host="0.0.0.0"` means it accepts connections from any network interface (not just localhost)

**Why uvicorn?**
FastAPI is just a framework — it needs a server to actually listen for HTTP requests. Uvicorn is that server. Think of FastAPI as the blueprint and uvicorn as the building.

---

### `server.py` — The API Layer (Waiter)

```python
from fastapi import FastAPI, Query
from .clients.rq_client import queue
from .queues.worker import process_query

app = FastAPI()

@app.get('/')
def root():
    return {"status": "Server is running up"}

@app.post('/chat')
def chat(
    query: str = Query(..., description="The chat query of the user")
):
    job = queue.enqueue(process_query, query)
    return {"status": "queued", "job_id": job.id}

@app.get('/job-status')
def get_result(
    job_id: str = Query(..., description="JOB ID")
):
    job = queue.fetch_job(job_id=job_id)
    result = job.return_value()
    return {"result": result}
```

**Three endpoints:**

| Endpoint | Method | What it does |
|---|---|---|
| `/` | GET | Health check — confirms server is alive |
| `/chat` | POST | Takes query, pushes job to queue, returns job_id immediately |
| `/job-status` | GET | Takes job_id, returns the result if job is done |

**Key insight — `/chat` returns instantly:**
```python
job = queue.enqueue(process_query, query)  # push to queue, doesn't wait
return {"status": "queued", "job_id": job.id}  # respond in milliseconds
```

The user gets a `job_id` immediately. The actual LLM work happens separately in the background.

**`Query(...)` — what is that?**
`Query` is a FastAPI utility that reads parameters from the URL query string.
- `...` means the parameter is required (not optional)
- `description` is shown in the auto-generated API docs at `/docs`

So a request looks like:
```
POST /chat?query=what+is+authentication
GET  /job-status?job_id=abc-123-xyz
```

---

### `clients/rq_client.py` — The Queue Connection (Order Rail)

```python
from redis import Redis
from rq import Queue

queue = Queue(connection=Redis(
    host="localhost",
    port="6379"
))
```

**What it does:**
- Connects to Valkey (which is Redis-compatible) running in Docker on port `6379`
- Creates an RQ `Queue` object that is used to push and pull jobs

**Why a separate file?**
So both `server.py` (which enqueues jobs) and potentially other files can import the same `queue` instance without creating multiple connections.

**What is Valkey?**
Valkey is an open-source fork of Redis. It works exactly like Redis — same protocol, same commands. RQ uses it as a message broker: a place to store pending jobs and their results.

---

### `queues/worker.py` — The RAG Logic (Chef)

```python
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def process_query(query: str):
    # Step 1: Connect to Qdrant
    vector_db = QdrantVectorStore.from_existing_collection(...)

    # Step 2: Search for relevant chunks
    search_result = vector_db.similarity_search(query=query)

    # Step 3: Build context string from chunks
    context = "\n\n\n".join([
        f"Page Content: {result.page_content}
         \nPage Number: {result.metadata.get('page')}
         \nPage Label: {result.metadata.get('page_label')}
         \nSource: {result.metadata.get('source')}"
        for result in search_result
    ])

    # Step 4: Call Gemini with context in system prompt
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )

    # Step 5: Return the answer
    return response.choices[0].message.content
```

**Why is `vector_db` created inside the function?**
If it were at the top of the file (module level), Python would try to connect to Qdrant the moment `server.py` imports this file — even before Qdrant is ready. Moving it inside the function means the connection only happens when a job actually runs.

**The 5-step RAG flow inside `process_query`:**

```
query: "what is authentication?"
    |
    | Step 1: Connect to Qdrant
    v
    | Step 2: similarity_search → finds 4 most relevant PDF chunks
    v
    | Step 3: Build context string with page content + page number + source
    v
    | Step 4: Send to Gemini:
    |   system: "You are a helpful assistant... here is context: {context}"
    |   user: "what is authentication?"
    v
    | Step 5: Return Gemini's answer
    v
"Authentication is the process of identifying who is using..."
```

---

### `docker-compose.yml` — Infrastructure

```yaml
services:
    valkey:
        image: valkey/valkey
        ports:
            - 6379:6379
```

Runs Valkey (Redis-compatible) in a Docker container. RQ needs this to store and retrieve jobs.

The `rag_agent` folder has a separate `docker-compose.yml` that runs Qdrant on port `6333`.

---

## How All Pieces Connect — Full Request Flow

```
1. Both Docker containers running:
   - Qdrant on :6333  (vector search)
   - Valkey on :6379  (job queue)

2. Server starts:
   python -m rag_queue.main
   → uvicorn listening on :8000

3. User sends request:
   POST /chat?query=what is authentication

4. server.py receives it:
   → queue.enqueue(process_query, "what is authentication")
   → Redis stores the job
   → Returns: {"status": "queued", "job_id": "abc-123"}

5. RQ worker (running separately) picks up the job:
   → runs process_query("what is authentication")
   → searches Qdrant → gets chunks
   → calls Gemini → gets answer
   → stores result back in Redis

6. User polls:
   GET /job-status?job_id=abc-123
   → Returns: {"result": "Authentication is the process of..."}
```

---

## Why This Design is Better Than Simple RAG

| Simple RAG (chat.py) | Queue RAG (rag_queue) |
|---|---|
| One user at a time | Many users simultaneously |
| Blocks until answer ready | Returns immediately with job_id |
| CLI only | HTTP API — any client can use it |
| Crashes under load | Workers handle jobs independently |
| No retry on failure | RQ can retry failed jobs |

---

## Running the Project

```bash
# Terminal 1: Start Qdrant
cd rag_agent
docker compose up -d

# Terminal 2: Start Valkey
cd rag_queue
docker compose up -d

# Terminal 3: Start RQ Worker (needed to process jobs)
cd /Users/techntactix/Documents/python-llm
rq worker --with-scheduler

# Terminal 4: Start the API server
cd /Users/techntactix/Documents/python-llm
python -m rag_queue.main
```

Then open `http://localhost:8000/docs` to see the interactive API documentation.

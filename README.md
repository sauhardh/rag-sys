# RAG with Interview Booking Assistant

A Retrieval-Augmented Generation (RAG) and conversational assistant built with FastAPI, PostgreSQL, Qdrant Vector DB, Redis, and OpenAI-compatible LLM providers (Groq / Ollama).

The system supports document ingestion (PDF & TXT), multi-strategy chunking, scoped vector search, conversational chat memory, intent detection, and automated interview scheduling with tool calling.

---

## Environment Configuration

Create a `.env` file in the root directory:

```bash
cp .env.example .env  # or create .env directly
```

### 1. Database & Cache Defaults (Shared)

```env
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password
POSTGRES_DB=rag_db
DATABASE_URL=postgresql+asyncpg://rag_user:rag_password@postgres:5432/rag_db

REDIS_URL=redis://redis:6379/0
```

> **Note for Manual/Local running**: Change the host in `DATABASE_URL` and `REDIS_URL` from `postgres` / `redis` to `localhost`.

---

### 2. LLM Provider Options

Choose **either** Groq or Ollama:

#### Option A: Groq (Recommended for fast cloud inference)

```env
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-20b
```

#### Option B: Local Ollama (Self-hosted)

```env
OPENAI_BASE_URL=http://host.docker.internal:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=mistral:latest
```

---

## How to Run

### Method 1: The Docker Way (Recommended)

Make sure Docker and Docker Compose are installed and running.

1. **Start all services:**

   ```bash
   docker compose up -d --build
   ```

2. **Check logs:**

   ```bash
   docker compose logs server -f
   ```

3. **Access Interactive API Docs:**
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

4. **Stop the stack:**

   ```bash
   docker compose down
   # To wipe stored data volumes (clean reset):
   docker compose down -v
   ```

---

### Method 2: The Manual / Local Way

#### Prerequisites

- Python 3.13+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Running instances of PostgreSQL, Qdrant (`localhost:6333`), and Redis (`localhost:6379`).

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Update `.env` hosts to `localhost`:**

   ```env
   DATABASE_URL=postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db
   REDIS_URL=redis://localhost:6379/0
   ```

3. **Run the server:**

   ```bash
   uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

---

## Special Notes for Ollama Users

If you are using host-installed Ollama with Docker:

1. **Bind Ollama to `0.0.0.0`:**
   By default, Ollama only listens on `127.0.0.1` and blocks container connections.
   Start Ollama with:

   ```bash
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

2. **Linux Firewall (`ufw`):**
   Allow incoming connections to the Ollama port:

   ```bash
   sudo ufw allow 11434/tcp
   ```

3. **Pull your target model:**

   ```bash
   ollama pull mistral:latest
   ```

---

## API Structure & Usage

### 1. Upload & Index Document

Upload a PDF or TXT file to extract text, create embeddings, and store them in Qdrant.

- **Endpoint:** `POST /api/v1/document`
- **Query Parameter:** `chunking_strategy` (`recursive` | `character`) - default: `recursive`
- **Body (form-data):** `file: <your_file.pdf>`

**Example Response:**

```json
{
  "status": "success",
  "payload": {
    "document_id": "c3e5e40e-7d84-486a-b286-9f44a8677c77",
    "filename": "annual_report.pdf",
    "chunking_strategy": "recursive",
    "chunk_count": 42,
    "embedding_dimension": 384,
    "status": "indexed"
  }
}
```

---

### 2. Chat & RAG Assistant

Engage in conversation, ask document-related questions, or book an interview.

- **Endpoint:** `POST /api/v1/chat`
- **Request Body (JSON):**

```json
{
  "session_id": "session-12345",
  "message": "What were the total sales mentioned in the report?",
  "document_id": "c3e5e40e-7d84-486a-b286-9f44a8677c77"
}
```

#### Fields to Notice

| Field         | Type                | Description                                                                                         |
| ------------- | ------------------- | --------------------------------------------------------------------------------------------------- |
| `session_id`  | `string` (required) | Unique session identifier for maintaining multi-turn memory in Redis.                               |
| `message`     | `string` (required) | The user message or question.                                                                       |
| `document_id` | `string` (optional) | The document UUID returned from the upload API. **Only chunks from this document will be queried.** |

#### Behavior

- **With `document_id` provided:** Performs contextual RAG search on the specific document and returns matching chunks in `sources`.
- **Without `document_id`:** Performs natural conversational AI without hallucinating document facts or returning phantom sources.
- **Interview Booking Intent:** If the user indicates intent to schedule an interview, the assistant switches to booking mode, collects name, email, date (`YYYY-MM-DD`), and time (`HH:MM`), and persists the confirmed booking into PostgreSQL (`booking_created: true`).

**Example Response:**

```json
{
  "session_id": "session-12345",
  "response": "The total sales reported for Q4 were $1.2M according to section 3.2.",
  "sources": ["annual_report.pdf"],
  "booking_created": false
}
```

# NutriGen AI — Personal Nutrition Assistant

> An AI-powered, RAG-augmented diet planning system that generates personalized 7-day Indian meal plans through a conversational interface.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage Guide](#usage-guide)

---

## Project Overview

NutriGen AI is an end-to-end conversational nutrition planning system. A user describes their health goals and lifestyle through a natural chat interface, and the system generates a fully personalized 7-day Indian meal plan grounded in real nutritional knowledge retrieved from PDFs and YouTube transcripts.

**Key capabilities:**

- Conversational profile collection — the agent asks only what it needs, one question at a time
- Automatic BMI, BMR, and TDEE calculation tailored to the user's activity level and goal
- RAG-based meal generation — meal suggestions are grounded in a curated nutrition knowledge base
- Health-condition-aware planning — handles diabetes, hypertension, PCOS, thyroid, lactose intolerance, and obesity
- Structured JSON diet plan output rendered as a clean, tabbed weekly view in Streamlit

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph (StateGraph with MemorySaver checkpointing) |
| LLM | DeepSeek Chat via OpenRouter API |
| Embeddings | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (HuggingFace) |
| Vector Store | ChromaDB (MMR retrieval, cosine similarity) |
| Data Ingestion | LangChain PDF Loader + YouTube Transcript API |
| Frontend | Streamlit |
| Environment | Python-dotenv, OpenAI-compatible SDK |

---

## Architecture

### System Overview (Prose)

The system is built as a stateful LangGraph pipeline. Every user message triggers the graph from the `START` node. The `profile_structurer` node parses the full conversation history and extracts a structured user profile. A routing function then checks whether the profile is complete — if not, the `input_collector` node generates the next targeted question and halts. Once complete, the graph proceeds through metric calculation, semantic query building, RAG retrieval, diet generation, and output formatting.

The knowledge base is built offline via a separate ingestion pipeline that loads PDFs and YouTube transcripts, chunks them, embeds them, and persists them to a local Chroma vector store. At inference time, the diet generator loads this store and retrieves the top-5 most relevant chunks using MMR search before passing them as context to the LLM.

### Graph Flow (ASCII Diagram)

```
START
  │
  ▼
┌─────────────────────┐
│  profile_structurer │  ← Parses full conversation, extracts structured profile
└─────────────────────┘
  │
  ├─── is_profile_complete = False ──► ┌─────────────────┐
  │                                    │ input_collector │ → asks next missing field → END
  │                                    └─────────────────┘
  │
  └─── is_profile_complete = True ───►
  │
  ▼
┌──────────────────────┐
│  metrics_calculator  │  ← Computes BMI, BMR, TDEE, calorie target
└──────────────────────┘
  │
  ▼
┌───────────────┐
│ query_builder │  ← Converts profile + metrics into semantic search query
└───────────────┘
  │
  ▼
┌────────────────┐
│ diet_generator │  ← Retrieves RAG context from Chroma, calls LLM
└────────────────┘
  │
  ▼
┌──────────────────┐
│ output_formatter │  ← Parses JSON, strips markdown fences, injects disclaimer
└──────────────────┘
  │
  ▼
 END
```

### Ingestion Pipeline (ASCII Diagram)

```
 PDFs (./data/)           YouTube Excel Sheet
      │                          │
      ▼                          ▼
 PyPDFLoader            YouTubeTranscriptApi
      │                          │
      └──────────┬───────────────┘
                 ▼
       RecursiveCharacterTextSplitter
       (chunk_size=1000, overlap=200)
                 │
                 ▼
     HuggingFace Embeddings (MiniLM)
                 │
                 ▼
        ChromaDB Vector Store
        (persisted to ./vector_store)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (for DeepSeek Chat)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nutrigen-ai.git
cd nutrigen-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 5. Build the knowledge base

**From PDFs** — place your nutrition PDFs inside a `./data/` directory, then run:

```bash
python -c "
from ingestion.pdf_loader import load_pdfs
from ingestion.splitter import chunk_documents
from ingestion.embedder import get_embeddings
from ingestion.vector_store import create_vector_store

docs = load_pdfs('./data')
chunks = chunk_documents(docs)
embeddings = get_embeddings()
create_vector_store(chunks, embeddings)
"
```

**From YouTube** — prepare an Excel file with columns `Title` and `link`, then run:

```bash
python -c "
from ingestion.youtube_loader import load_youtube_from_excel
from ingestion.splitter import chunk_documents
from ingestion.embedder import get_embeddings
from ingestion.vector_store import create_vector_store

docs = load_youtube_from_excel('./data/youtube_links.xlsx')
chunks = chunk_documents(docs)
embeddings = get_embeddings()
create_vector_store(chunks, embeddings)
"
```

> Both sources can be combined — simply concatenate the document lists before chunking.

### 6. Run the application

```bash
streamlit run app.py
```

---

## Usage Guide

1. Open the app in your browser link: [live link](https://nutrigenai-akzhrhnzaytxgzrbf2ttn2.streamlit.app/)
2. The assistant greets you and asks for your primary goal
3. Answer each question naturally — the agent collects your age, weight, height, goal, workout habits, sleep quality, and any health conditions one at a time
4. Once all fields are collected, the agent automatically calculates your metrics and generates a personalized 7-day meal plan
5. The plan is displayed as a tabbed weekly view — one tab per day, with breakfast, lunch, dinner, and snacks
6. Your profile and health metrics are visible in the sidebar throughout the session
7. Use **Start New Session** in the sidebar to reset and begin a fresh plan


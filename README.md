
# 🍳 Recipe Helper

An AI-powered recipe assistant that uses **Retrieval-Augmented Generation (RAG)** to answer cooking questions based on real recipe data crawled from the web.

Ask questions like *"How do I make pasta carbonara?"* or *"What can I cook with chicken and rice?"* and get answers backed by actual recipe sources.

## How It Works

```
┌──────────────────────────────────────────────────────┐
│                   INGESTION (run once)                │
│                                                      │
│  Recipe Websites ──► Crawl ──► Chunk ──► Pinecone DB │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                  CHAT APP (run anytime)               │
│                                                      │
│  User Question ──► Search Pinecone ──► AI Answer     │
│                                        + Sources     │
└──────────────────────────────────────────────────────┘
```

**Ingestion Pipeline** crawls recipe websites using Tavily, splits the content into chunks, converts them to vector embeddings, and stores them in Pinecone.

**Chat App** takes a user's question, searches Pinecone for the most relevant recipe chunks, and uses an LLM to generate an answer with cited sources.

## Tech Stack

- **LangChain** — RAG pipeline and agent orchestration
- **Ollama** (Mistral) — local LLM for generating answers
- **Pinecone** — vector database for storing and searching recipe embeddings
- **Tavily** — web crawling to collect recipe data
- **Streamlit** — chat interface
- **Python** — backend logic

## Project Structure

```
Recipe-helper/
├── main.py              # Streamlit chat interface
├── ingestion.py         # Crawls recipe sites and stores data in Pinecone
├── logger.py            # Colored terminal logging
├── backend/
│   └── core.py          # RAG pipeline — retrieval + LLM answer generation
├── .env                 # API keys (not in repo)
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed with Mistral model (`ollama pull mistral`)
- Pinecone account (free tier works)
- Tavily API key

### Installation

```bash
git clone https://github.com/Javicilie0/Recipe-helper.git
cd Recipe-helper
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=your_index_name
TAVILY_API_KEY=your_tavily_api_key
```

### Run

**Step 1:** Ingest recipe data (run once)

```bash
python ingestion.py
```

**Step 2:** Start the chat app

```bash
streamlit run main.py
```

## Features

- Async web crawling of multiple recipe sites simultaneously
- Batch document indexing for reliable Pinecone uploads
- Source citations with every answer
- Chat history within session
- Clear chat functionality

# Multi-Agent RAG System

This repository implements a **production-style Multi-Agent Retrieval-Augmented Generation (RAG)** system for legal documents using **LangChain**, **LangGraph**, and a modular service architecture.

The system supports:
- Asynchronous ingestion of legal PDFs
- Hybrid retrieval (BM25 + vector search)
- Multi-agent reasoning with critique and caching
- Containerized deployment with Docker

---

## Architecture Overview

The system is composed of **three main layers**:

1. **Ingestion Layer** – Parses, chunks, embeds, and indexes documents  
2. **Retrieval Layer** – Hybrid retrieval using BM25 + dense vectors  
3. **Reasoning Layer** – Multi-agent orchestration using LangGraph  

---

## Agent Roles

### 1. Answer Agent
**Responsibility**
- Generates a legal answer to the user query
- Uses retrieved legal passages via tools
- Produces a structured answer with citations

**Key characteristics**
- Does not see internal user identifiers
- Calls retrieval tools when needed
- Optimized for factual grounding

---

### 2. Critic Agent
**Responsibility**
- Reviews the Answer Agent’s output
- Approves or rejects based on:
  - Grounding in retrieved sources
  - Hallucination risk
  - Answer completeness

**Outcomes**
- `approve` → answer is cached
- `reject` → answer is retried (bounded retries)

---

### 3️⃣ Tooling Layer (Implicit Agent)
**Responsibility**
- Executes retrieval and search operations
- Enforces user-level document isolation
- Interfaces with vector DB and BM25 index

This layer is **not visible to the LLM** and operates using system context.

---

## LangGraph Flow

The orchestration is implemented using **LangGraph**, enabling deterministic, inspectable agent execution.

### Flow Diagram (Logical)

```
User Query
    ↓
Cache Check
    ↓
Answer Agent
    ↓
Critic Agent
    ├── approve → Cache → Return Answer
    └── reject  → Retry (bounded) → Answer Agent
```

---

## Retrieval Approach

### Hybrid Retrieval Strategy

The system combines:

1. **BM25 (OpenSearch)**
   - High recall for exact legal phrasing
   - Strong on statute names and citations

2. **Vector Search (Qdrant)**
   - Semantic matching
   - Handles paraphrased legal queries

Results are merged, deduplicated, reranked using a cross-encoder, and truncated to top-K passages.

---

## Ingestion Pipeline

1. PDF loading and text repair  
2. Boilerplate detection and removal  
3. Hierarchical chunking:
   - Parent chunks (context)
   - Child chunks (retrieval units)
4. Embedding generation  
5. Indexing into:
   - Qdrant (vectors)
   - OpenSearch (BM25)

Ingestion runs **asynchronously** with job tracking via Redis.

---

## Tech Stack

| Component | Technology |
|--------|-----------|
| API | FastAPI |
| Orchestration | LangGraph |
| LLM | Gemini |
| Vector DB | Qdrant |
| BM25 | OpenSearch |
| Cache | Redis |
| Database | PostgreSQL |
| Embeddings | Sentence-Transformers |
| Deployment | Docker / Docker Compose |

---

## Running the System

### Prerequisites
- Docker
- Docker Compose

### Setup

```bash
git clone <repo-url>
cd legal-ai
cp .env.example .env
```
add you model api in docker compose api environment variables
Fill in required API keys in `docker_compose`.

### Start Services

```bash
docker compose up --build
```

### Access

| Service | URL |
|------|-----|
| API Docs | http://localhost:8000/docs |
| Qdrant UI | http://localhost:6333/dashboard |
| OpenSearch | http://localhost:9200 |

---

## API Endpoints (Key)

- `POST /ingest/pdf` – Upload and ingest legal PDFs  
- `GET /ingest/status/{job_id}` – Track ingestion progress  
- `POST /search` – Hybrid document retrieval  
- `POST /qa` – End-to-end multi-agent question answering  
- `POST /evaluation` - evaluation on the dataset

---

## Design Decisions & Rationale

### Why Multi-Agent?
- Separates reasoning from verification
- Reduces hallucinations
- Enables explainable failures

### Why Hybrid Retrieval?
- Legal queries require exact matching **and** semantic understanding
- BM25 alone fails on paraphrases
- Vector search alone misses precise citations

### Why Dockerized?
- Reproducibility
- Easy reviewer evaluation
- Clear service boundaries


# Evaluation Report

**Run ID:** `eval_20260122_111007`  
**Status:** Completed  
**Dataset:** sample  

---

## Progress Summary

- **Total Queries:** 8  
- **Processed Queries:** 8  
- **Completion:** 100%  
- **Current Status:** completed  

---

## Overall Results

| Metric | Value |
|------|------|
| Total Queries | 8 |
| Valid Evaluations | 8 |
| Passed | 8 |
| Failed | 0 |
| Pass Rate | 100% |
| Avg Retrieval Latency (ms) | 351.16 |
| Avg Generation Latency (ms) | 3794.27 |
| Avg Total Latency (ms) | 5963.78 |

---

## Answer Quality Metrics

| Metric | Average Score |
|------|---------------|
| Faithfulness | 1.00 |
| Relevance | 1.00 |
| Citation Accuracy | 0.75 |
| Completeness | 0.96 |
| Hallucination Rate | 0.00 |

---

## Category Breakdown

| Category | Total | Passed | Pass Rate |
|--------|------|--------|-----------|
| Securities Fraud | 4 | 4 | 100% |
| Risk Disclosure | 2 | 2 | 100% |
| Disclosure Requirements | 1 | 1 | 100% |
| Omission Liability | 1 | 1 | 100% |

---

## Summary

- All queries passed successfully with **no hallucinations detected**.
- Strong performance across **faithfulness** and **relevance** metrics.
- **Citation accuracy (0.75)** indicates room for improvement in source attribution.
- Latency metrics are consistent with multi-step RAG pipelines.


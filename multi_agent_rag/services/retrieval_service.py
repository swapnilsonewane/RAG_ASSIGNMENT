import os
import time
import asyncio
import logging
from typing import List, Dict

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import ResponseHandlingException

from sentence_transformers import CrossEncoder
from opensearchpy import AsyncOpenSearch

from prometheus_client import Counter, Histogram

# ======================================================
# CONFIG
# ======================================================

QDRANT_URL = os.environ.get("QDRANT_URL")
if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL must be set in container environment")

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legal_chunks")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "legal_chunks_bm25")

TOP_K_VECTOR = 10
TOP_K_BM25 = 10
FINAL_K = 6

RERANK_BATCH_SIZE = 16
RERANK_TIMEOUT = 2.0

MAX_RETRIES = 3
RETRY_DELAY = 2.0
QDRANT_TIMEOUT = 60.0

# ======================================================
# LOGGING
# ======================================================

logger = logging.getLogger("retrieval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ======================================================
# METRICS
# ======================================================

REQ_TOTAL = Counter("legal_retrieval_requests_total", "Total retrieval requests")
REQ_FAILURE = Counter("legal_retrieval_failures_total", "Retrieval failures")
QDRANT_RETRIES = Counter("qdrant_retries_total", "Total Qdrant retry attempts")
QDRANT_TIMEOUTS = Counter("qdrant_timeouts_total", "Total Qdrant timeouts")

LATENCY = {
    "bm25": Histogram("retrieval_bm25_latency_seconds", "BM25 latency"),
    "vector": Histogram("retrieval_vector_latency_seconds", "Vector latency"),
    "rerank": Histogram("retrieval_rerank_latency_seconds", "Rerank latency"),
    "total": Histogram("retrieval_total_latency_seconds", "Total latency"),
}

# ======================================================
# CLIENTS
# ======================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

qdrant = QdrantClient(
    url=QDRANT_URL,
    timeout=QDRANT_TIMEOUT,
)

opensearch = AsyncOpenSearch(hosts=[OPENSEARCH_URL])

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ======================================================
# SCHEMA
# ======================================================

class RetrievedPassage(BaseModel):
    text: str
    citation: str
    score: float


class RetrievalResult(BaseModel):
    passages: List[RetrievedPassage]


class QueryLegalDocsInput(BaseModel):
    query: str = Field(..., description="User legal question")

# ======================================================
# UTIL
# ======================================================

def build_citation(meta: Dict) -> str:
    """Build citation from metadata"""
    chunk_id = meta.get('chunk_id', '')
    chunk_ref = chunk_id[:12] if chunk_id else meta.get('chunk_hash', '')[:12]

    case_info = []
    if meta.get('case_name'):
        case_info.append(f"Case: {meta['case_name']}")
    
    case_info.extend([
        f"Source: {meta.get('source_file', 'Unknown')}",
        f"Page {meta.get('page_number', 'N/A')}",
    ])
    
    return " | ".join(case_info)


async def retry_with_backoff(func, max_retries=MAX_RETRIES, initial_delay=RETRY_DELAY):
    """Retry a function with exponential backoff"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except (ResponseHandlingException, TimeoutError, asyncio.TimeoutError, Exception) as e:
            last_exception = e
            
            if "timed out" in str(e).lower():
                QDRANT_TIMEOUTS.inc()
            
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                QDRANT_RETRIES.inc()
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {str(e)[:200]}")
    
    raise last_exception

# ======================================================
# SEARCH PRIMITIVES
# ======================================================

async def bm25_search(query: str, user_id: str) -> List[Document]:
    """BM25 keyword search using OpenSearch"""
    start = time.perf_counter()

    try:
        resp = await opensearch.search(
            index=OPENSEARCH_INDEX,
            body={
                "size": TOP_K_BM25,
                "query": {
                    "bool": {
                        "must": [{"match": {"text": query}}],
                        "filter": [
                            {"term": {"chunk_type": "child"}},
                            {"term": {"user_id": user_id}},
                        ],
                    }
                },
            },
        )

        LATENCY["bm25"].observe(time.perf_counter() - start)
        
        hits = resp["hits"]["hits"]
        logger.info(f"BM25 search returned {len(hits)} results")

        return [
            Document(
                page_content=h["_source"]["text"],
                metadata=h["_source"],
            )
            for h in hits
        ]
    
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}, returning empty results")
        LATENCY["bm25"].observe(time.perf_counter() - start)
        return []


async def vector_search(query: str, user_id: str) -> List[Document]:
    """Semantic vector search using Qdrant v1.11.3"""
    start = time.perf_counter()

    try:

        vector = await asyncio.to_thread(
            embeddings.embed_query,
            query,
        )
        
        logger.debug(f"Generated embedding vector of dimension {len(vector)}")

        async def search_with_retry():
            def search_qdrant():

                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="chunk_type",
                            match=MatchValue(value="child"),
                        ),
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id),
                        ),
                    ]
                )

                response = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    query_filter=query_filter,
                    limit=TOP_K_VECTOR,
                    with_payload=True,
                )
                
                return response
            
            return await asyncio.to_thread(search_qdrant)

        response = await retry_with_backoff(search_with_retry)

        results = response.points if hasattr(response, 'points') else []
        
        LATENCY["vector"].observe(time.perf_counter() - start)
        logger.info(f"Vector search returned {len(results)} results")

        documents = []
        for r in results:
            if hasattr(r, 'payload') and 'text' in r.payload:
                documents.append(Document(
                    page_content=r.payload["text"],
                    metadata=r.payload,
                ))
            else:
                logger.warning(f"Result missing payload or text field")
        
        return documents

    except Exception as e:
        logger.exception(f"Vector search failed after all retries: {e}")
        LATENCY["vector"].observe(time.perf_counter() - start)
        return []


# ======================================================
# RERANK
# ======================================================

async def rerank(query: str, docs: List[Document]) -> List[Document]:
    """Rerank documents using cross-encoder"""
    start = time.perf_counter()

    if not docs:
        logger.warning("No documents to rerank")
        return []

    pairs = [(query, d.page_content) for d in docs]

    try:
        scores = await asyncio.wait_for(
            asyncio.to_thread(
                reranker.predict,
                pairs,
                batch_size=RERANK_BATCH_SIZE,
            ),
            timeout=RERANK_TIMEOUT,
        )

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        LATENCY["rerank"].observe(time.perf_counter() - start)

        final_docs = []
        for doc, score in ranked[:FINAL_K]:
            doc.metadata['rerank_score'] = float(score)
            final_docs.append(doc)
        
        logger.info(f"Reranked {len(docs)} docs, returning top {FINAL_K}")
        return final_docs

    except asyncio.TimeoutError:
        logger.warning("Reranker timeout; using fallback order")
        return docs[:FINAL_K]
    except Exception as e:
        logger.error(f"Reranking failed: {e}, using original order")
        return docs[:FINAL_K]


# ======================================================
# PIPELINE
# ======================================================

async def retrieve(query: str, user_id: str) -> RetrievalResult:
    """Main retrieval pipeline: BM25 + Vector + Rerank"""
    REQ_TOTAL.inc()
    start = time.perf_counter()

    logger.info(f"Starting retrieval for query='{query[:50]}...' user_id={user_id}")

    try:

        bm25_docs, vector_docs = await asyncio.gather(
            bm25_search(query, user_id),
            vector_search(query, user_id),
        )
        
        logger.info(f"Retrieved: {len(bm25_docs)} BM25, {len(vector_docs)} vector")

        dedup: Dict[str, Document] = {}
        for doc in bm25_docs + vector_docs:
            key = doc.metadata.get("chunk_id") or doc.metadata.get("chunk_hash")
            if key:
                if key not in dedup:
                    dedup[key] = doc
            else:
                logger.warning(f"Document missing chunk_id: {doc.metadata.get('source_file')}")

        if not dedup:
            logger.warning(f"No results found for query: {query[:50]}...")
            return RetrievalResult(passages=[])

        logger.info(f"Found {len(dedup)} unique chunks before reranking")

        ranked = await rerank(query, list(dedup.values()))

        passages = [
            RetrievedPassage(
                text=d.page_content.strip(),
                citation=build_citation(d.metadata),
                score=d.metadata.get('rerank_score', 0.0),
            )
            for d in ranked
        ]

        logger.info(f"Returning {len(passages)} passages after reranking")
        return RetrievalResult(passages=passages)

    except Exception as e:
        REQ_FAILURE.inc()
        logger.exception(f"Retrieval failed for query: {query[:50]}...")
        raise

    finally:
        LATENCY["total"].observe(time.perf_counter() - start)


# ======================================================
# TOOL INTERFACE
# ======================================================

async def query_legal_docs(query: str) -> dict:
    """Wrapper for agent tool usage - uses hardcoded user_id"""
    result = await retrieve(query, user_id="viewer")
    return result.model_dump()
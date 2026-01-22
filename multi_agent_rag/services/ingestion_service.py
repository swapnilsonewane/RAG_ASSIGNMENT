import os
import re
import uuid
import hashlib
import asyncio
import logging
from datetime import datetime
from collections import Counter
from typing import List, Optional, Dict, Tuple

import redis.asyncio as redis
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    VectorParams,
    Distance,
)
from multi_agent_rag.services.opensearch_sync import OpenSearchSyncService

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = os.getenv("DATA_DIR", "./data")
QDRANT_URL = os.getenv("QDRANT_URL")
REDIS_URL = os.getenv("REDIS_URL")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legal_chunks")

EMBED_BATCH_SIZE = 32
EMBED_CONCURRENCY = 4
MAX_RETRIES = 3

PARENT_CHUNK_SIZE = 1500
PARENT_OVERLAP = 300
CHILD_CHUNK_SIZE = 500
CHILD_OVERLAP = 100

if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL must be set")

if not REDIS_URL:
    raise RuntimeError("REDIS_URL must be set")

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("legal_ingestion")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ============================================================
# SCHEMAS
# ============================================================

class IngestResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    documents_processed: int
    chunks_indexed: int
    error: Optional[str]
    updated_at: str


# ============================================================
# ADAPTIVE TEXT CLEANING
# ============================================================

class AdaptiveTextCleaner:
    """
    Adaptive text cleaning that learns patterns from the document
    rather than using hard-coded rules.
    """
    
    def __init__(self, min_line_length: int = 10):
        self.min_line_length = min_line_length
    
    def repair_hyphenation(self, text: str) -> str:
        """Fix words broken across lines with hyphens"""
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure"""
        text = text.replace('\t', ' ')
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def detect_and_remove_boilerplate(
        self, 
        pages: List[Document], 
        frequency_threshold: float = 0.5
    ) -> Tuple[set, Dict[str, int]]:
        """
        Adaptively detect boilerplate text by analyzing repetition patterns.
        Returns both the boilerplate set and frequency statistics.
        """
        line_counter = Counter()
        total_pages = len(pages)
        
        for page in pages:
            for line in page.page_content.split('\n'):

                normalized = ' '.join(line.strip().lower().split())
                if len(normalized) >= self.min_line_length:
                    line_counter[normalized] += 1
        
        threshold = int(total_pages * frequency_threshold)
        
        boilerplate = {
            line for line, count in line_counter.items() 
            if count >= threshold
        }
        
        stats = {
            'total_unique_lines': len(line_counter),
            'boilerplate_lines': len(boilerplate),
            'threshold_used': threshold
        }
        
        logger.info(f"Detected {len(boilerplate)} boilerplate patterns "
                   f"from {len(line_counter)} unique lines "
                   f"(threshold: {threshold} occurrences)")
        
        return boilerplate, stats
    
    def remove_boilerplate_from_text(
        self, 
        text: str, 
        boilerplate: set
    ) -> Tuple[str, List[str]]:
        """Remove boilerplate lines while preserving document structure"""
        clean_lines = []
        removed_lines = []
        
        for line in text.split('\n'):
            normalized = ' '.join(line.strip().lower().split())
            
            if normalized in boilerplate:
                removed_lines.append(line)
            else:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip(), removed_lines
    
    def clean_document(self, text: str) -> str:
        """Complete cleaning pipeline"""
        text = self.repair_hyphenation(text)

        text = self.normalize_whitespace(text)
        
        return text


# ============================================================
# CONTEXT-AWARE METADATA EXTRACTION
# ============================================================

class MetadataExtractor:
    """Extract metadata from legal documents adaptively"""
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extract legal citations using flexible patterns"""
        citations = []

        pattern = r'\b\d+\s+U\.?\s?S\.?\s+\d+\b'
        citations.extend(re.findall(pattern, text))
        
        pattern = r'\b\d+\s+F\.\s?\d*[d]?\s+\d+\b'
        citations.extend(re.findall(pattern, text))
        
        pattern = r'\bNo\.\s+\d+-\d+\b'
        citations.extend(re.findall(pattern, text))
        
        return list(set(citations))
    
    @staticmethod
    def extract_case_info(text: str) -> Dict[str, str]:
        """Extract case information from document header"""
        metadata = {}

        header = text[:2000]

        case_pattern = r'([A-Z][A-Za-z\s,\.&]+)\s+v\.?\s+([A-Z][A-Za-z\s,\.&]+)'
        match = re.search(case_pattern, header)
        if match:
            metadata['plaintiff'] = match.group(1).strip()
            metadata['defendant'] = match.group(2).strip()
            metadata['case_name'] = f"{match.group(1).strip()} v. {match.group(2).strip()}"

        docket_pattern = r'No\.\s+(\d+-\d+)'
        match = re.search(docket_pattern, header)
        if match:
            metadata['docket_number'] = match.group(1)

        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, header)
        if years:
            metadata['year'] = years[0]
        
        return metadata


# ============================================================
# INTELLIGENT CHUNKING STRATEGY
# ============================================================

class LegalDocumentChunker:
    """
    Context-aware chunking that preserves legal document structure
    and semantic coherence.
    """
    
    def __init__(
        self,
        parent_size: int = PARENT_CHUNK_SIZE,
        child_size: int = CHILD_CHUNK_SIZE,
        parent_overlap: int = PARENT_OVERLAP,
        child_overlap: int = CHILD_OVERLAP,
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
            ],
            length_function=len,
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
            ],
            length_function=len,
        )
    
    def create_hierarchical_chunks(
        self,
        documents: List[Document],
        boilerplate: set,
        cleaner: AdaptiveTextCleaner,
    ) -> List[Document]:
        """
        Create parent-child chunk hierarchy with preserved context.
        """
        all_chunks = []
        
        for doc in documents:

            cleaned_text = cleaner.clean_document(doc.page_content)
            
            cleaned_text, removed = cleaner.remove_boilerplate_from_text(
                cleaned_text, boilerplate
            )
            
            if not cleaned_text.strip():
                logger.warning(f"Document page {doc.metadata.get('page_number')} "
                             f"empty after cleaning")
                continue
            
            doc.page_content = cleaned_text
            
            parent_docs = self.parent_splitter.split_documents([doc])
            
            for parent_idx, parent in enumerate(parent_docs):

                parent_content_hash = hashlib.sha256(
                    parent.page_content.encode('utf-8')
                ).hexdigest()[:16]
                
                parent_id = f"{doc.metadata.get('doc_id', 'unknown')}_{parent_content_hash}"
                parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, parent_id))
                
                parent.metadata.update({
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                    "chunk_id": parent_uuid,
                    "parent_index": parent_idx,
                    "char_count": len(parent.page_content),
                    "word_count": len(parent.page_content.split()),
                    "ingested_at": datetime.utcnow().isoformat(),
                })
                
                all_chunks.append(parent)
                
                child_docs = self.child_splitter.split_documents([parent])
                
                for child_idx, child in enumerate(child_docs):

                    child_content_hash = hashlib.sha256(
                        child.page_content.encode('utf-8')
                    ).hexdigest()[:16]
                    
                    child_id = f"{parent_id}_child_{child_idx}_{child_content_hash}"
                    child_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, child_id))
                    
                    child.metadata.update({
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "chunk_id": child_uuid,
                        "child_index": child_idx,
                        "parent_chunk_id": parent_uuid,
                        "char_count": len(child.page_content),
                        "word_count": len(child.page_content.split()),
                        "ingested_at": datetime.utcnow().isoformat(),
                    })
                    
                    all_chunks.append(child)
        
        logger.info(f"Created {len(all_chunks)} total chunks from "
                   f"{len(documents)} pages")
        
        return all_chunks


# ============================================================
# INGESTION SERVICE
# ============================================================

class LegalIngestionService:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.qdrant = QdrantClient(url=QDRANT_URL)

        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        
        self.opensearch_sync = OpenSearchSyncService()
        
        self.cleaner = AdaptiveTextCleaner()
        self.chunker = LegalDocumentChunker()
        self.metadata_extractor = MetadataExtractor()
        
        self._embed_semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)
        self._embedding_dim = len(
            self.embeddings.embed_documents(["dimension_probe"])[0]
        )

    async def close(self) -> None:
        await self.redis.close()
        await self.opensearch_sync.close()

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    async def _ensure_collection(self) -> None:
        """Ensure Qdrant collection exists - using sync client"""
        collections_response = await asyncio.to_thread(
            self.qdrant.get_collections
        )
        existing = {c.name for c in collections_response.collections}

        if COLLECTION_NAME not in existing:
            logger.info("Creating Qdrant collection: %s", COLLECTION_NAME)
            await asyncio.to_thread(
                self.qdrant.create_collection,
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def load_documents(self, pdf_path: str, user_id: str) -> List[Document]:
        """Load PDF and enrich with initial metadata"""
        pages = PyPDFLoader(pdf_path).load()
        doc_id = self._hash(pdf_path)
        filename = os.path.basename(pdf_path)

        documents: List[Document] = []

        for page in pages:

            page.page_content = self.cleaner.clean_document(page.page_content)
            
            if page.metadata.get("page") == 0 or page.metadata.get("page") == 1:
                case_metadata = self.metadata_extractor.extract_case_info(
                    page.page_content
                )
            else:
                case_metadata = {}
            
            citations = self.metadata_extractor.extract_citations(
                page.page_content
            )

            page.metadata.update({
                "doc_id": doc_id,
                "user_id": user_id,
                "source_file": filename,
                "page_number": page.metadata.get("page", 0),
                "doc_type": "legal_document",
                "citations": citations,
                **case_metadata,
            })
            
            documents.append(page)

        logger.info(f"Loaded {len(documents)} pages from {filename}")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents with adaptive chunking"""

        boilerplate, stats = self.cleaner.detect_and_remove_boilerplate(documents)
        
        logger.info(f"Boilerplate detection stats: {stats}")

        chunks = self.chunker.create_hierarchical_chunks(
            documents, 
            boilerplate, 
            self.cleaner
        )

        valid_chunks = []
        for chunk in chunks:

            if len(chunk.page_content.strip()) < 50:
                logger.debug(f"Skipping short chunk: {len(chunk.page_content)} chars")
                continue

            chunk.metadata['chunk_quality_score'] = self._calculate_chunk_quality(
                chunk.page_content
            )
            
            valid_chunks.append(chunk)
        
        logger.info(f"Created {len(valid_chunks)} valid chunks "
                   f"({len(chunks) - len(valid_chunks)} filtered out)")
        
        return valid_chunks
    
    def _calculate_chunk_quality(self, text: str) -> float:
        """Calculate a simple quality score for the chunk"""
        score = 0.0

        length = len(text)
        if 200 <= length <= 800:
            score += 0.4
        elif 100 <= length < 200 or 800 < length <= 1200:
            score += 0.2

        if text.strip()[-1] in '.!?':
            score += 0.3

        words = text.split()
        if len(words) >= 20:
            score += 0.3
        
        return min(score, 1.0)

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        async with self._embed_semaphore:
            return await asyncio.to_thread(
                self.embeddings.embed_documents,
                texts,
            )

    async def _upsert_chunks(self, chunks: List[Document], job_id: str) -> None:
        """Index chunks to both Qdrant and OpenSearch"""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                for i in range(0, len(chunks), EMBED_BATCH_SIZE):
                    batch = chunks[i:i + EMBED_BATCH_SIZE]

                    vectors = await self._embed_batch(
                        [c.page_content for c in batch]
                    )

                    points = [
                        PointStruct(
                            id=c.metadata["chunk_id"],
                            vector=vectors[idx],
                            payload={
                                "text": c.page_content,
                                **c.metadata,
                            },
                        )
                        for idx, c in enumerate(batch)
                    ]

                    await asyncio.to_thread(
                        self.qdrant.upsert,
                        collection_name=COLLECTION_NAME,
                        points=points,
                    )

                    opensearch_docs = [
                        {
                            "text": c.page_content,
                            **c.metadata,
                        }
                        for c in batch
                    ]
                    await self.opensearch_sync.index_documents(opensearch_docs)
                    
                    logger.info(f"Indexed batch {i//EMBED_BATCH_SIZE + 1} "
                              f"({len(batch)} chunks)")

                return

            except Exception as exc:
                logger.error(
                    "Job %s | upsert attempt %d failed: %s",
                    job_id,
                    attempt,
                    exc,
                    exc_info=True,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Qdrant upsert failed after retries")

    async def run_ingest_job(self, job_id: str, pdf_path: str, user_id: str) -> None:
        """Main ingestion pipeline"""
        try:
            await self.redis.hset(job_id, mapping={
                "status": "running",
                "updated_at": datetime.utcnow().isoformat(),
            })

            await self._ensure_collection()

            logger.info(f"Job {job_id}: Loading document {pdf_path}")
            documents = self.load_documents(pdf_path, user_id)

            logger.info(f"Job {job_id}: Chunking {len(documents)} pages")
            chunks = self.chunk_documents(documents)

            logger.info(f"Job {job_id}: Indexing {len(chunks)} chunks")
            await self._upsert_chunks(chunks, job_id)

            await self.redis.hset(job_id, mapping={
                "status": "completed",
                "documents_processed": len(documents),
                "chunks_indexed": len(chunks),
                "updated_at": datetime.utcnow().isoformat(),
            })
            
            logger.info(f"Job {job_id}: Completed successfully")

        except Exception as exc:
            logger.exception(f"Job {job_id}: Failed with error: {exc}")
            await self.redis.hset(job_id, mapping={
                "status": "failed",
                "error": str(exc),
                "updated_at": datetime.utcnow().isoformat(),
            })
            raise

    async def create_job(
        self,
        document_id: str,
        pdf_path: str,
        user_id: str,
    ) -> str:
        """Create and start an ingestion job"""
        job_id = str(uuid.uuid4())

        await self.redis.hset(job_id, mapping={
            "job_id": job_id,
            "document_id": document_id,
            "user_id": user_id,
            "status": "queued",
            "documents_processed": 0,
            "chunks_indexed": 0,
            "error": "",
            "updated_at": datetime.utcnow().isoformat(),
        })

        asyncio.create_task(
            self.run_ingest_job(job_id, pdf_path, user_id)
        )

        return job_id
import os
import logging
import asyncio
from typing import List, Dict, Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import ScrollRequest

# ======================================================
# CONFIG
# ======================================================

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legal_chunks")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "legal_chunks_bm25")

BATCH_SIZE = 100

# ======================================================
# LOGGING
# ======================================================

logger = logging.getLogger("opensearch_sync")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ======================================================
# OPENSEARCH SYNC SERVICE
# ======================================================

class OpenSearchSyncService:
    def __init__(self):
        self.opensearch = AsyncOpenSearch(hosts=[OPENSEARCH_URL])
        self.qdrant = AsyncQdrantClient(url=QDRANT_URL)

    async def close(self):
        await self.opensearch.close()
        await self.qdrant.close()

    async def ensure_index(self):
        """Create OpenSearch index if it doesn't exist"""
        try:
            exists = await self.opensearch.indices.exists(index=OPENSEARCH_INDEX)
            if exists:
                logger.info(f"OpenSearch index '{OPENSEARCH_INDEX}' already exists")
                return

            index_body = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "legal_analyzer": {
                                "type": "standard",
                                "stopwords": "_english_"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "legal_analyzer"
                        },
                        "chunk_id": {"type": "keyword"},
                        "chunk_type": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "parent_id": {"type": "keyword"},
                        "source_file": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "doc_type": {"type": "keyword"},
                        "child_index": {"type": "integer"},
                        "ingested_at": {"type": "date"}
                    }
                }
            }

            await self.opensearch.indices.create(
                index=OPENSEARCH_INDEX,
                body=index_body
            )
            logger.info(f"Created OpenSearch index: {OPENSEARCH_INDEX}")

        except Exception as e:
            logger.error(f"Failed to ensure index: {e}")
            raise

    async def sync_from_qdrant(self, user_id: str = None):
        """Sync all documents from Qdrant to OpenSearch"""
        try:
            await self.ensure_index()

            offset = None
            total_synced = 0

            while True:

                scroll_result = await self.qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=BATCH_SIZE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, next_offset = scroll_result

                if not points:
                    break

                if user_id:
                    points = [p for p in points if p.payload.get("user_id") == user_id]

                if not points:
                    offset = next_offset
                    continue

                bulk_body = []
                for point in points:

                    bulk_body.append({
                        "index": {
                            "_index": OPENSEARCH_INDEX,
                            "_id": point.payload.get("chunk_id", str(point.id))
                        }
                    })

                    bulk_body.append(point.payload)

                if bulk_body:
                    response = await self.opensearch.bulk(body=bulk_body)
                    
                    if response.get("errors"):
                        logger.warning(f"Some documents failed to index: {response}")
                    
                    total_synced += len(points)
                    logger.info(f"Synced {total_synced} documents to OpenSearch")

                if next_offset is None:
                    break

                offset = next_offset

            logger.info(f"Sync complete. Total documents synced: {total_synced}")
            return total_synced

        except Exception as e:
            logger.error(f"Failed to sync from Qdrant: {e}")
            raise

    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index a batch of documents to OpenSearch"""
        try:
            await self.ensure_index()

            if not documents:
                return

            bulk_body = []
            for doc in documents:
                chunk_id = doc.get("chunk_id")
                if not chunk_id:
                    logger.warning("Document missing chunk_id, skipping")
                    continue

                bulk_body.append({
                    "index": {
                        "_index": OPENSEARCH_INDEX,
                        "_id": chunk_id
                    }
                })
                bulk_body.append(doc)

            if bulk_body:
                response = await self.opensearch.bulk(body=bulk_body)
                
                if response.get("errors"):
                    logger.warning(f"Some documents failed to index: {response}")
                else:
                    logger.info(f"Indexed {len(documents)} documents to OpenSearch")

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise

    async def delete_by_user(self, user_id: str):
        """Delete all documents for a specific user"""
        try:
            response = await self.opensearch.delete_by_query(
                index=OPENSEARCH_INDEX,
                body={
                    "query": {
                        "term": {"user_id": user_id}
                    }
                }
            )
            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} documents for user {user_id}")
            return deleted

        except NotFoundError:
            logger.info(f"No documents found for user {user_id}")
            return 0
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
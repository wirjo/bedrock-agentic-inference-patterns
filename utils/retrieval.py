"""One retrieval interface for Bedrock Knowledge Bases and local Chroma.

Selection is configuration-based. A configured Knowledge Base never silently
falls back after an error. Local indexing happens during construction, outside
the measured query; Chroma's default embedding model may download on first use.
"""

from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path

from . import workshop


class KnowledgeBaseRetriever:
    """Retrieve from the configured Knowledge Base using the existing AWS client."""

    backend = "Amazon Bedrock Knowledge Bases"

    def __init__(self, settings):
        """Retain the Region, Knowledge Base ID, and timeout defaults."""
        self.settings = settings

    def retrieve(self, query, result_limit=3, timeout=60):
        """Return native retrieval results and propagate service errors."""
        with closing(workshop.make_retrieval_client(self.settings, timeout)) as client:
            return client.retrieve(
                knowledgeBaseId=self.settings.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": result_limit},
                },
            )


class ChromaRetriever:
    """Persist and search the short demo policy documents with local embeddings."""

    backend = "Local Chroma"

    def __init__(self, directory=None, documents=None, embedding_function=None):
        """Index changed text files and metadata before any measured retrieval.

        The demo uses one chunk per short policy. Longer production documents
        need an explicit chunking strategy. embedding_function is injectable for
        offline tests; normal use selects Chroma's ONNX MiniLM embedding model.
        """
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        directory = directory or os.getenv(
            "BEDROCK_CHROMA_PATH", str(workshop.ROOT / ".chroma/notebook")
        )
        documents = Path(documents or workshop.ROOT / "data/knowledge-base")
        self.client = chromadb.PersistentClient(
            path=str(directory), settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            "support-policies",
            embedding_function=embedding_function or DefaultEmbeddingFunction(),
        )
        existing = self.collection.get(include=["metadatas"])
        fingerprints = {
            key: (metadata or {}).get("_fingerprint")
            for key, metadata in zip(existing["ids"], existing["metadatas"])
        }
        current = []
        for path in sorted(documents.glob("*.txt")):
            text = path.read_text().strip()
            if not text:
                continue
            current.append(path.name)
            metadata_path = path.with_name(path.name + ".metadata.json")
            metadata = {}
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text()).get("metadataAttributes", {})
            fingerprint = sha256((text + json.dumps(metadata, sort_keys=True)).encode()).hexdigest()
            if fingerprints.get(path.name) == fingerprint:
                continue
            self.collection.upsert(
                ids=[path.name], documents=[text],
                metadatas=[{**metadata, "_fingerprint": fingerprint, "source": path.name}],
            )
        removed = sorted(set(fingerprints) - set(current))
        if removed:
            self.collection.delete(ids=removed)

    def retrieve(self, query, result_limit=3, timeout=60):
        """Return text hits in the same envelope as the Knowledge Base backend.

        timeout is accepted for interface compatibility. Local embedding/query
        work is synchronous and is not forcibly interrupted by a network timeout.
        """
        if result_limit < 1:
            raise ValueError("result_limit must be positive")
        count = self.collection.count()
        if count == 0:
            return {"retrievalResults": []}
        found = self.collection.query(
            query_texts=[query], n_results=min(count, result_limit),
            include=["documents", "metadatas"],
        )
        results = []
        for text, metadata in zip(found["documents"][0], found["metadatas"][0]):
            results.append({
                "content": {"type": "TEXT", "text": text},
                "location": {
                    "type": "CUSTOM",
                    "customDocumentLocation": {"id": metadata["source"]},
                },
                "metadata": metadata,
            })
        return {"retrievalResults": results}


def create_retriever(settings, **local_options):
    """Use Knowledge Bases when its ID is set; otherwise prepare local Chroma."""
    if settings.knowledge_base_id:
        return KnowledgeBaseRetriever(settings)
    return ChromaRetriever(**local_options)

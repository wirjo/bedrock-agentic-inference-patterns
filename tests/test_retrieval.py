from pathlib import Path

import pytest
from chromadb.api.types import EmbeddingFunction
from botocore.exceptions import ClientError

from utils import retrieval
from utils.workshop import Settings


class DeterministicEmbedding(EmbeddingFunction):
    """Small deterministic vectors for storage/query tests; not semantic benchmarks."""

    def __init__(self):
        pass

    def __call__(self, input):
        return [
            [float("replacement" in text.lower()), float("warranty" in text.lower()), 0.1]
            for text in input
        ]

    @staticmethod
    def name():
        return "support-policy-test-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return DeterministicEmbedding()


def test_no_knowledge_base_uses_real_chroma_and_syncs_changed_documents(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "delivery.txt").write_text("A replacement policy for delayed orders.")
    (documents / "warranty.txt").write_text("Warranty terms for defective products.")
    options = dict(
        directory=tmp_path / "index", documents=documents, embedding_function=DeterministicEmbedding(),
    )
    retriever = retrieval.create_retriever(Settings(knowledge_base_id=""), **options)
    assert retriever.backend == "Local Chroma"
    result = retriever.retrieve("replacement", result_limit=1)
    assert result["retrievalResults"][0]["location"]["customDocumentLocation"]["id"] == "delivery.txt"
    (documents / "delivery.txt").write_text("Updated replacement policy.")
    (documents / "warranty.txt").unlink()
    updated = retrieval.create_retriever(Settings(knowledge_base_id=""), **options)
    result = updated.retrieve("replacement", result_limit=5)
    assert len(result["retrievalResults"]) == 1
    assert result["retrievalResults"][0]["content"]["text"] == "Updated replacement policy."


def test_empty_local_corpus_returns_no_evidence(tmp_path):
    documents = tmp_path / "empty"
    documents.mkdir()
    retriever = retrieval.ChromaRetriever(
        directory=tmp_path / "index", documents=documents, embedding_function=DeterministicEmbedding(),
    )
    assert retriever.retrieve("replacement") == {"retrievalResults": []}


def test_configured_knowledge_base_never_silently_falls_back(monkeypatch):
    error = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "Retrieve")

    class DeniedClient:
        def retrieve(self, **kwargs):
            raise error

        def close(self):
            pass

    def forbidden_chroma(*args, **kwargs):
        raise AssertionError("A configured Knowledge Base must not fall back")

    monkeypatch.setattr(retrieval, "ChromaRetriever", forbidden_chroma)
    monkeypatch.setattr(retrieval.workshop, "make_retrieval_client", lambda *args: DeniedClient())
    retriever = retrieval.create_retriever(Settings(knowledge_base_id="KBTEST1234"))
    assert retriever.backend == "Amazon Bedrock Knowledge Bases"
    with pytest.raises(ClientError, match="AccessDenied"):
        retriever.retrieve("replacement")

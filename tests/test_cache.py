from unittest.mock import AsyncMock

import pytest
from conftest import request_for
from qdrant_client import AsyncQdrantClient

from gateway.cache.scope import build_cache_key
from gateway.cache.semantic_cache import SemanticCache
from gateway.cache.vector_store import VectorStore, payload_filter
from gateway.router.model_router import ModelRouter


def make_key(config, request, tier=None):
    return build_cache_key(request, ModelRouter(config).select(request, tier), config)


def test_scope_separates_model_options_and_source(config):
    request = request_for()
    key = make_key(config, request)
    assert key.scope_id != make_key(config, request, "premium").scope_id
    assert key.scope_id != make_key(config, request.model_copy(update={"temperature": 0.0})).scope_id
    assert key.scope_id != make_key(config, request_for("Make this email more polite: Different facts.")).scope_id
    assert key.scope_id == make_key(config, request.model_copy(update={"stream": True})).scope_id


def test_exact_unknown_extraction_and_history(config):
    one = make_key(config, request_for("Extract the invoice number: INV-1"))
    two = make_key(config, request_for("Extract the invoice number: INV-2"))
    assert one.exact and two.exact
    assert one.filters != two.filters
    assert make_key(config, request_for("What is 2+2?")).exact
    request = request_for("Continue.")
    request.messages.insert(0, request.messages[0].model_copy(update={"role": "system", "content": "Policy A"}))
    before = make_key(config, request)
    request.messages[0].content = "Policy B"
    assert before.scope_id != make_key(config, request).scope_id


def test_output_constraints_are_exact(config):
    one = make_key(config, request_for("Summarize this text in two bullets: Same source."))
    two = make_key(config, request_for("Summarize this text in three bullets: Same source."))
    assert one.scope_id != two.scope_id
    a = make_key(config, request_for("Rewrite this text to sound polite: Same source."))
    b = make_key(config, request_for("Rewrite this text to sound professional: Same source."))
    assert a.scope_id != b.scope_id


def test_no_unscoped_queries():
    with pytest.raises(ValueError):
        payload_filter({})


@pytest.mark.asyncio
async def test_qdrant_local_filtering_and_legacy_exclusion(config):
    # Real Qdrant client in in-memory mode, no Docker/network needed.
    store = VectorStore.__new__(VectorStore)
    store.client = AsyncQdrantClient(location=":memory:")
    store.collection_name = "queries"
    store.vector_size = 384
    await store.ensure_collection()
    key = make_key(config, request_for())
    vector = [1.0] + [0.0] * 383
    try:
        await store.upsert(vector, {**key.fields, "response": "local", "model": "phi4-mini"})
        assert (await store.search(vector, 0.95, key.filters))["response"] == "local"
        premium = make_key(config, request_for(), "premium")
        assert await store.search(vector, 0.95, premium.filters) is None
        assert await store.search([-x for x in vector], 0.95, key.filters) is None
        from qdrant_client.models import PointStruct
        await store.client.upsert("queries", [PointStruct(id=1, vector=vector, payload={"response": "legacy"})])
        assert await store.search(vector, 0.95, premium.filters) is None

        exact = make_key(config, request_for("Extract the total: $12"))
        await store.upsert([0.0] * 384, {**exact.fields, "response": "$12"})
        assert (await store.find_exact(exact.filters))["response"] == "$12"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_exact_cache_skips_embedding_and_protects_scope(config):
    class NoEmbedding:
        def embed(self, text):
            raise AssertionError("Exact lookup must not embed")

    store = AsyncMock()
    store.vector_size = 384
    store.find_exact.return_value = None
    cache = SemanticCache(NoEmbedding(), store)
    key = make_key(config, request_for("Extract the total: $12"))
    assert await cache.lookup(key) is None
    await cache.store(key, "$12", {"scope_id": "malicious"})
    assert store.upsert.call_args.kwargs["payload"]["scope_id"] == key.scope_id

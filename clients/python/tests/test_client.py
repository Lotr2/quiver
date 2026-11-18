"""Integration tests for the Python client. They need a running server:

    quiver-server --port 7070 --data-dir /tmp/quiver-test &
    QUIVER_URL=http://localhost:7070 pytest clients/python/tests
"""

import os
import random
import uuid

import pytest

from quiver_client import Quiver, QuiverError

URL = os.environ.get("QUIVER_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set QUIVER_URL to run against a live server")


@pytest.fixture
def collection():
    db = Quiver(URL)
    name = "t-" + uuid.uuid4().hex[:12]
    col = db.create_collection(name, dim=8, metric="l2", sync="flush")
    yield col
    db.drop_collection(name)


def random_points(n, dim=8, seed=0):
    rng = random.Random(seed)
    return [
        {"id": f"p{i}", "vector": [rng.gauss(0, 1) for _ in range(dim)], "metadata": {"bucket": i % 4}}
        for i in range(n)
    ]


def test_health():
    assert Quiver(URL).health()["status"] == "ok"


def test_upsert_search_get_delete(collection):
    points = random_points(500)
    assert collection.upsert(points, batch_size=128) == 500

    result = collection.search(points[7]["vector"], k=3)
    assert result.hits[0].id == "p7"
    assert result.plan["strategy"] == "graph"

    filtered = collection.search(points[7]["vector"], k=10, filter={"bucket": 2})
    assert all(h.metadata["bucket"] == 2 for h in filtered)

    assert collection.get("p7")["metadata"] == {"bucket": 3}
    assert collection.delete(["p7", "missing"]) == 1
    assert collection.get("p7") is None
    assert collection.info()["points"] == 499


def test_errors_are_typed(collection):
    with pytest.raises(QuiverError) as err:
        collection.search([1.0, 2.0], k=3)
    assert err.value.status == 400
    assert err.value.code == "invalid_argument"

    with pytest.raises(QuiverError) as err:
        Quiver(URL).collection("does-not-exist").info()
    assert err.value.status == 404


def test_compaction_and_checkpoint(collection):
    collection.upsert(random_points(100))
    collection.delete([f"p{i}" for i in range(50)])
    info = collection.compact()
    assert info["tombstones"] == 0 and info["points"] == 50
    assert collection.checkpoint()["wal_bytes"] == 0

# quiver-client

A dependency-free Python client for the [Quiver](../../README.md) vector database. It uses only the standard library.

```bash
pip install ./clients/python
```

```python
from quiver_client import Quiver

db = Quiver("http://127.0.0.1:7070")
docs = db.create_collection("docs", dim=3, metric="cosine", exist_ok=True)

docs.upsert([
    {"id": "a", "vector": [1, 0, 0], "metadata": {"lang": "en"}},
    {"id": "b", "vector": [0, 1, 0], "metadata": {"lang": "de"}},
])

result = docs.search([0.9, 0.1, 0], k=5, filter={"lang": "en"})
for hit in result:
    print(hit.id, hit.score, hit.metadata)
print(result.plan)   # {'strategy': 'exact_scan', 'candidates': 1, ...}
```

Errors come back as `QuiverError`, which carries the HTTP `status`, a machine-readable `code` and a `message`.

"""A small, dependency-free Python client for the Quiver REST API.

    from quiver_client import Quiver

    db = Quiver("http://127.0.0.1:7070")
    docs = db.create_collection("docs", dim=384, metric="cosine")
    docs.upsert([{"id": "a", "vector": [...], "metadata": {"lang": "en"}}])
    hits = docs.search([...], k=5, filter={"lang": "en"})
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

__all__ = ["Quiver", "Collection", "Hit", "SearchResult", "QuiverError"]
__version__ = "0.1.0"


class QuiverError(Exception):
    """An error response from the server, carrying its HTTP status and error code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    vector: Optional[list] = None


@dataclass
class SearchResult:
    hits: list
    plan: dict
    took_ms: float

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)


class Quiver:
    """Connection to a Quiver server."""

    def __init__(self, url: str = "http://127.0.0.1:7070", timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    # -- transport --------------------------------------------------------------
    def _request(self, method: str, path: str, body: Any = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                ctype = resp.headers.get("Content-Type", "")
                return json.loads(payload) if ctype.startswith("application/json") else payload.decode()
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read())["error"]
                raise QuiverError(e.code, err.get("code", "error"), err.get("message", "")) from None
            except (ValueError, KeyError):
                raise QuiverError(e.code, "error", str(e)) from None

    # -- server -----------------------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/health")

    def metrics(self) -> str:
        return self._request("GET", "/metrics")

    # -- collections ------------------------------------------------------------
    def create_collection(
        self,
        name: str,
        dim: int,
        metric: str = "cosine",
        *,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        sync: str = "fsync",
        exist_ok: bool = False,
    ) -> "Collection":
        config = {
            "name": name,
            "dim": dim,
            "metric": metric,
            "M": M,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "sync": sync,
        }
        try:
            self._request("POST", "/collections", config)
        except QuiverError as e:
            if not (exist_ok and e.status == 409):
                raise
        return Collection(self, name)

    def collection(self, name: str) -> "Collection":
        return Collection(self, name)

    def list_collections(self) -> list:
        return self._request("GET", "/collections")["collections"]

    def drop_collection(self, name: str) -> None:
        self._request("DELETE", f"/collections/{urllib.parse.quote(name)}")


class Collection:
    """Handle to one collection; cheap to create, holds no server state."""

    def __init__(self, client: Quiver, name: str):
        self._client = client
        self.name = name
        self._base = f"/collections/{urllib.parse.quote(name)}"

    def info(self) -> dict:
        return self._client._request("GET", self._base)

    def upsert(self, points: Iterable[dict], batch_size: int = 1000) -> int:
        """Insert or replace points ({"id", "vector", "metadata"}), sent in batches.
        Each batch is atomic on the server."""
        total, batch = 0, []
        for p in points:
            vector = p["vector"]
            batch.append({**p, "vector": vector.tolist() if hasattr(vector, "tolist") else list(vector)})
            if len(batch) >= batch_size:
                total += self._client._request("POST", f"{self._base}/points", {"points": batch})["upserted"]
                batch = []
        if batch:
            total += self._client._request("POST", f"{self._base}/points", {"points": batch})["upserted"]
        return total

    def get(self, point_id: str) -> Optional[dict]:
        try:
            return self._client._request("GET", f"{self._base}/points/{urllib.parse.quote(str(point_id), safe='')}")
        except QuiverError as e:
            if e.status == 404:
                return None
            raise

    def delete(self, ids: Sequence[str]) -> int:
        return self._client._request("POST", f"{self._base}/points/delete", {"ids": [str(i) for i in ids]})["deleted"]

    def search(
        self,
        vector: Sequence[float],
        k: int = 10,
        *,
        ef: Optional[int] = None,
        filter: Optional[dict] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
    ) -> SearchResult:
        body: dict = {
            "vector": vector.tolist() if hasattr(vector, "tolist") else list(vector),
            "k": k,
            "include_vectors": include_vectors,
            "include_metadata": include_metadata,
        }
        if ef is not None:
            body["ef"] = ef
        if filter:
            body["filter"] = filter
        r = self._client._request("POST", f"{self._base}/search", body)
        hits = [Hit(h["id"], h["score"], h.get("metadata", {}), h.get("vector")) for h in r["hits"]]
        return SearchResult(hits, r["plan"], r["took_ms"])

    def checkpoint(self) -> dict:
        return self._client._request("POST", f"{self._base}/checkpoint", {})

    def compact(self) -> dict:
        return self._client._request("POST", f"{self._base}/compact", {})

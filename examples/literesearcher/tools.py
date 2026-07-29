"""LLM-facing search/fetch tools + harness-only logging helpers.

Search and fetch make HTTP calls to the stateless retrieval server.

The per-fetch character cap is enforced server-side from
``serve.yaml`` (``fetch_max_chars``).
"""

import httpx

_NOT_IN_SESSION_MSG = (
    "URL {url!r} was not returned by any prior /search call in this session. "
    "Either fix the URL formatting (Wikipedia titles use underscores, e.g. "
    "'Foo_Bar'; percent-encode special characters), or pick a URL from a "
    "previous search result."
)

page_ids = {}


def search(query: str, limit: int = 10, source: str | None = None) -> dict:
    """Hybrid search over Wikipedia. Returns {results: [{url, title, snippet}]}."""
    base_url = "http://localhost:8000"
    source = "browsecomp_plus"
    payload = {"query": query, "limit": limit}
    if source:
        payload["source"] = source
    r = httpx.post(
        f"{base_url}/search",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    hits = r.json()["results"]

    pages = [
        {
            "url": h["url"],
            "page_id": h["page_id"],
            "title": h["title"],
            "snippet": h["snippet"],
        }
        for h in hits
    ]
    for page in pages:
        page_ids[page["url"]] = page["page_id"]
    public = [
        {"url": h["url"], "title": h["title"], "snippet": h["snippet"]} for h in hits
    ]
    return {"results": public}


def fetch(url: str, query: str, source: str | None = None) -> dict:
    """Full body of a previously-searched URL.

    Returns ``{content, truncated, event_id}`` or ``{error, message, event_id}``.
    """
    base_url = "http://localhost:8000"
    source = "browsecomp_plus"
    if not (url in page_ids):
        result = {
            "error": "url_not_in_session",
            "message": _NOT_IN_SESSION_MSG.format(url=url),
        }
        return result
    payload = {"page_id": page_ids[url]}
    if source:
        payload["source"] = source
    r = httpx.post(
        f"{base_url}/fetch",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    result = r.json()
    return result

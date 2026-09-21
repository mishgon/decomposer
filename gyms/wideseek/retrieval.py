"""CPU E5 + upstream Qdrant index, with the upstream /retrieve and /access API."""
import argparse
import json
from pathlib import Path
import sys
import threading

from fastapi import FastAPI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
import torch
import uvicorn


class Query(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=8)
    topk: int = Field(default=3, ge=1, le=10)


class Access(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=8)


def serve(assets, upstream, port, qdrant_url):
    sys.path.insert(0, str(upstream / "examples/agent/tools/search_local_server_qdrant"))
    from qdrant_encoder import Encoder
    torch.set_num_threads(4)
    encoder = Encoder("e5", str(assets / "e5-base-v2"), "mean", 256, False, torch.device("cpu"))
    client = QdrantClient(url=qdrant_url, timeout=120)
    collection = "wiki_collection_m32_cef512"
    client.get_collection(collection)
    pages = {}
    with (assets / "Wiki-2018-Corpus/wiki_webpages.jsonl").open() as stream:
        for index, line in enumerate(stream):
            page = json.loads(line)
            pages[page["url"]] = page
            if index % 100000 == 0:
                print(f"Loaded {index} pages", flush=True)
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ready", "pages": len(pages), "collection": collection,
                "encoder_device": "cpu", "encoder_dtype": "float32",
                **json.loads((assets / "assets.json").read_text())}

    @app.post("/retrieve")
    def retrieve(request: Query):
        with lock:
            vectors = encoder.encode(request.queries)
        result = [client.query_points(collection, query=v.tolist(), limit=request.topk,
                    search_params=models.SearchParams(hnsw_ef=256), with_payload=True).points for v in vectors]
        return {"result": [[p.payload for p in points] for points in result]}

    @app.post("/access")
    def access(request: Access):
        return {"result": [pages.get(u.replace("index.php/", "index.php?title=")) for u in request.urls]}

    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("external/RLinf"))
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16333")
    args = parser.parse_args()
    serve(args.assets, args.upstream, args.port, args.qdrant_url)

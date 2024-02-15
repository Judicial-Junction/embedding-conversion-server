from fastapi import FastAPI
from opensearchpy import OpenSearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from models import Query
import uvicorn
import torch
import os

load_dotenv()
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
app = FastAPI()
client = OpenSearch(
    hosts=[{"host": os.getenv("OPENSEARCH_HOST", "0.0.0.0"), "port": 9200}],
    http_auth=(
        os.getenv("OPENSEARCH_USER", "admin"),
        os.getenv("OPENSEARCH_PASSWORD", "admin"),
    ),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    timeout=60,
)


@app.get("/")
async def health():
    return {"message": "Healthy"}


@app.post("/semantic_similarity")
async def semantic_similarity(request: Query):
    with torch.no_grad():
        mean_pooled = model.encode(request.message)

    query = {
        "size": 5,
        "query": {"knn": {"embedding": {"vector": mean_pooled, "k": 3}}},
        "_source": False,
        "fields": [
            "Case Number",
            "Case Title",
            "Judgement Date",
            "Judgement PDF URL",
            "Judgement Text",
        ],
    }

    response = client.search(body=query, index="case-text")
    return response["hits"]["hits"]


@app.post("/sentence_similarity")
async def sentence_similarity(request: Query):
    with torch.no_grad():
        mean_pooled = model.encode(request.message)

    query = {
        "size": 5,
        "query": {"knn": {"embedding": {"vector": mean_pooled, "k": 3}}},
        "_source": False,
        "fields": [
            "Judgement PDF URL",
            "Case Number",
            "Case Title",
            "Judgement Date",
            "Sentence",
        ],
    }

    response = client.search(body=query, index="sentence")
    return response["hits"]["hits"]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

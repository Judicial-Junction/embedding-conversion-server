from fastapi import FastAPI
from opensearchpy import OpenSearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from models import Query
import uvicorn
import torch
import boto3
import pickle
import json
import os

BIN_DIR = "bin/"

load_dotenv()
s3 = boto3.client("s3")
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
)

if os.path.isdir(BIN_DIR):
    os.mkdir("BIN_DIR")
    bucket_content = s3.list_objects(Bucket="sentence-index")
    for obj in bucket_content["Contents"]:
        file_name = os.path.join(BIN_DIR, obj["key"])
        s3.download_file("sentence-index", obj["key"], file_name)


with open("index.pkl", "rb") as file_handle:
    index = pickle.load(file_handle)

with open("case_uid_to_case_info.json", "rb") as file_handle:
    case_uid_to_case_info = json.load(file_handle)

with open("uid_to_sentence_mapping.json", "rb") as file_handle:
    uid_to_sentence_mapping = json.load(file_handle)


@app.get("/health")
async def health():
    return {"message": "Healthy"}


@app.get("/word_similarity")
async def word_similarity(request: Query):
    with torch.no_grad():
        mean_pooled = model.encode(request.message)

    query = {
        "size": 5,
        "query": {"knn": {"embedding": {"vector": mean_pooled, "k": 2}}},
        "_source": False,
        "fields": ["Case Number", "Case Title", "Judgement Date", "Judgement PDF URL"],
    }

    response = client.search(body=query, index="word_embedding")
    return response["hits"]["hits"]


@app.get("/sentence_similarity")
async def sentence_similarity(request: Query):
    query_embedding = model.encode([request.message])
    result, uid = index.search(query_embedding, k=3)
    return "hi"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

if not os.path.isdir(BIN_DIR):
    os.mkdir(BIN_DIR)
    bucket_content = s3.list_objects(Bucket="sentence-index")
    for obj in bucket_content["Contents"]:
        object_name = obj["Key"]
        file_name = os.path.join(BIN_DIR, object_name)
        s3.download_file("sentence-index", object_name, file_name)


with open(BIN_DIR+"index.pkl", "rb") as file_handle:
    index = pickle.load(file_handle)

with open(BIN_DIR+"case_uid_to_case_info.json", "rb") as file_handle:
    case_uid_to_case_info = json.load(file_handle)

with open(BIN_DIR+"uid_to_sentence_mapping.json", "rb") as file_handle:
    uid_to_sentence_mapping = json.load(file_handle)


@app.get("/health")
async def health():
    return {"message": "Healthy"}


@app.post("/word_similarity")
async def word_similarity(request: Query):
    with torch.no_grad():
        mean_pooled = model.encode(request.message)

    query = {
        "size": 5,
        "query": {"knn": {"embedding": {"vector": mean_pooled, "k": 3}}},
        "_source": False,
        "fields": ["Case Number", "Case Title", "Judgement Date", "Judgement PDF URL"],
    }

    response = client.search(body=query, index="word_embedding")
    return response["hits"]["hits"]


@app.post("/sentence_similarity")
async def sentence_similarity(request: Query):
    query_embedding = model.encode([request.message])
    response = []
    _ , uid = index.search(n=1, x=query_embedding, k=3)

    for n, i in enumerate(uid[0]):
        mini_response = {}
        case_no = str(i)[-5:]
        case_sent = str(i)[:-5]
        case_info = case_uid_to_case_info[str(int(case_no))]
        mini_response['ResultNumber'] = n+1
        mini_response['CaseNumber'] = case_info["c_no"]
        mini_response["CaseTitle"] = case_info["c_t"]
        mini_response["JudgementDate"] = case_info["j_d"]
        mini_response["PdfUrl"] = case_info["pdf"]
        mini_response["CaseText"] = case_sent
        mini_response["SimilarSentence"] = uid_to_sentence_mapping[str(i)]
        response.append(mini_response)

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

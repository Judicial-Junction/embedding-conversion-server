from fastapi import FastAPI
from opensearchpy import OpenSearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from models import Query
import uvicorn
import torch
import os
import json

load_dotenv()

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("model loaded")

with open("case_uid_to_case_info.json", "rb") as file_handle:
    case_uid_to_case_info = json.load(file_handle)

with open("uid_to_sentence_mapping.json", "rb") as file_handle:
    uid_to_sentence_mapping = json.load(file_handle)

print("mapping json loaded")

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
            "UID",
        ],
    }

    res = []

    response = client.search(body=query, index="sentence")
    result = response["hits"]["hits"]

    for i in result:
        mini_res = {}
        case_no = str(i["fields"]["UID"][0])[-5:]
        case_sent = str(i["fields"]["UID"][0])[:-5]
        case_info = case_uid_to_case_info[str(int(case_no))]
        mini_res["_index"] = i["_index"]
        mini_res["_id"] = i["_id"]
        mini_res["_score"] = i["_score"]
        mini_res["fields"] = {}
        mini_res["fields"]["Sentences"] = []
        mini_res["fields"]["Case Number"] = [case_info["c_no"]]
        mini_res["fields"]["Case Title"] = [case_info["c_t"]]
        mini_res["fields"]["Judgement Date"] = [case_info["j_d"]]
        mini_res["fields"]["Judgement PDF URL"] = [case_info["pdf"]]
        mini_res["fields"]["Sentences"].append(
            uid_to_sentence_mapping[str(int(case_sent) - 1) + case_no]
        )
        mini_res["fields"]["Sentences"].append(
            uid_to_sentence_mapping[str(i["fields"]["UID"][0])]
        )
        mini_res["fields"]["Sentences"].append(
            uid_to_sentence_mapping[str(int(case_sent) + 1) + case_no]
        )
        res.append(mini_res)

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from opensearchpy import OpenSearch
from dotenv import load_dotenv
from models import Query
import pandas as pd
import uvicorn
import os
import json

load_dotenv()

with open("case_uid_to_case_info.json", "rb") as file_handle:
    case_uid_to_case_info = json.load(file_handle)

with open("uid_to_sentence_mapping.json", "rb") as file_handle:
    uid_to_sentence_mapping = json.load(file_handle)

df = pd.read_csv('cases.csv')

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


@app.post("/sentence_similarity")
async def sentence_similarity(request: Query):
    query = {
        "_source": {
            "excludes": [
                "Sentence_embedding"
            ]
        },
        "query": {"neural": {"Sentence_embedding": {"query_text": request.message, "k": 5, "model_id": "O7sGY48BxYep_-iYPhhF"}}},
        "size": 5
    }

    res = []

    response = client.search(body=query, index="sentence")
    result = response["hits"]["hits"]["_source"]

    for i in result:
        mini_res = {}
        case_no = str(i["fields"]["CaseUID"][0])[-5:]
        case_sent = str(i["fields"]["CaseUID"][0])[:-5]
        case_info = case_uid_to_case_info[str(int(case_no))]
        mini_res["_index"] = i["_index"]
        mini_res["_id"] = i["_id"]
        mini_res["_score"] = i["_score"]
        mini_res["fields"] = {}
        mini_res["fields"]["Sentences"] = []
        mini_res["fields"]["Case Number"] = case_info["c_no"]
        mini_res["fields"]["Case Title"] = case_info["c_t"]
        mini_res["fields"]["Judgement Date"] = case_info["j_d"]
        mini_res["fields"]["Judgement PDF URL"] = case_info["pdf"]
        mini_res["fields"]["Judgement Text"] = df.loc[df['Case Number'] == case_info["c_no"], 'Judgement Text'].values[0]
        mini_res["fields"]["Sentences"].append(
            uid_to_sentence_mapping[str(int(case_sent) - 2) + case_no]
        )
        mini_res["fields"]["Sentences"].append(
            uid_to_sentence_mapping[str(int(case_sent) - 1) + case_no]
        )
        mini_res["fields"]["Sentences"].append(
            uid_to_sentence_mapping[str(i["fields"]["CaseUID"][0])]
        )
        mini_res["fields"]["Sentences"].append(
            uid_to_sentence_mapping[str(int(case_sent) + 1) + case_no]
        )
        mini_res["fields"]["Sentences"].append(
            uid_to_sentence_mapping[str(int(case_sent) + 2) + case_no]
        )
        res.append(mini_res)

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

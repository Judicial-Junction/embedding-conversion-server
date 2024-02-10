from fastapi import FastAPI
from opensearchpy import OpenSearch
from dotenv import load_dotenv
import os

load_dotenv()
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


@app.get("/health")
async def health():
    return {"message": "Healthy"}

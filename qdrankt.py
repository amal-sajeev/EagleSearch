from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://c8abf992-e97b-4ccd-a2b0-046e5c9f5ee5.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="iL-BfHumRrC_Ph8rOWLpGPT1JZu-8W6zfxxe1wm5cPaUiUeX3QSevg",
)

print(qdrant_client.get_collections())
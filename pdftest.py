from eaglesearch import Eaglesearch

cruncher = PDFProcessor(qdrant_url ="https://c8abf992-e97b-4ccd-a2b0-046e5c9f5ee5.europe-west3-0.gcp.cloud.qdrant.io",qdrant_api_key= "iL-BfHumRrC_Ph8rOWLpGPT1JZu-8W6zfxxe1wm5cPaUiUeX3QSevg", collection_name="pdftest")

# cruncher.ingest_pdf("C:/Users/User 3/Downloads/soul_calibur.pdf", batch_size=5)

hits = cruncher.search("Mitsurugi",limit=2)

for hit in hits:
    with open("C:/Users/User 3/Documents/Python scratchpad/pdf page search/output.txt", "wb") as out:
        out.write(str(hit.payload))
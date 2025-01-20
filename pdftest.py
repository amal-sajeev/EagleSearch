import io
from tqdm import tqdm
from PIL import Image



from eaglesearch import EagleSearch

cruncher = EagleSearch(qdrant_url ="https://c8abf992-e97b-4ccd-a2b0-046e5c9f5ee5.europe-west3-0.gcp.cloud.qdrant.io",qdrant_api_key= "iL-BfHumRrC_Ph8rOWLpGPT1JZu-8W6zfxxe1wm5cPaUiUeX3QSevg", collection_name="pdftest")

# cruncher.ingest_pdf("C:/Users/User 3/Downloads/Raj Kamal - Embedded Systems Architecture Programming and Design (Scanned Copy)-The McGraw Hill Companies-1-13.pdf", batch_size=5)

hits = cruncher.search("Components diagram",limit=3,score = True)

# with open("output.txt","w") as out:
#     out.write(str(hits))


n=0
payload = []

for i in hits.keys():
    payload.append(hits[i])
with open("output.txt", "w") as out:
    out.write(str(payload))
for i in tqdm(hits):
    cruncher.save_image(i,f"{n}.png")
    n+=1


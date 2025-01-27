import io
from tqdm import tqdm
from PIL import Image
import fitz


from eaglesearch import EagleSearch

cruncher = EagleSearch(qdrant_url ="https://c8abf992-e97b-4ccd-a2b0-046e5c9f5ee5.europe-west3-0.gcp.cloud.qdrant.io",qdrant_api_key= "iL-BfHumRrC_Ph8rOWLpGPT1JZu-8W6zfxxe1wm5cPaUiUeX3QSevg")

# cruncher.ingest_pdf("C:/Users/User 3/Downloads/grassugo.pdf", batch_size=5,collection_name="qwenufo")

# pdf = fitz.open("C:/Users/User 3/Downloads/Payslip_Dec_2024.pdf")
# with open("C:/Users/User 3/Downloads/Payslip_Dec_2024.pdf", "rb") as pdoc:
#     cruncher.ingest_pdf(pdoc,batch_size=5,collection_name="payslip")

hits = cruncher.search("pie charts",limit=1, prefetch_limit= 100, collection_name="payslip")

# with open("output.txt","a") as out:
#     out.write(str(hits))


n=0
payload = []

for i in tqdm(hits):
    with open("output.txt", "a") as out:
        out.write(f"{i["doc_id"]} : {i["score"]}\n")
    cruncher.save_image(i["metadata"]["page_image"],f"{n}.png")
    n+=1 
    
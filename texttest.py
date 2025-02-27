from eaglesearch import EagleSearch
import pprint
from io import BytesIO
from fastapi import UploadFile
import os
from tqdm import tqdm

txtchunker = EagleSearch(
    max_chunk_size=1500,
    similarity_threshold= 0.5,
    qdrant_api_key= "iL-BfHumRrC_Ph8rOWLpGPT1JZu-8W6zfxxe1wm5cPaUiUeX3QSevg",
    qdrant_url ="https://c8abf992-e97b-4ccd-a2b0-046e5c9f5ee5.europe-west3-0.gcp.cloud.qdrant.io"
    )

#JSON TEST

# with open("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/movie-database.json", "rb") as file:

#     # print(BytesIO(file.read()).read())

#     upfile = UploadFile(
#         file = BytesIO(file.read()),
#         filename = file.name,
#         size = os.path.getsize("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/movie-database.json")
#     )

# nuchunks = txtchunker.chunk_document(upfile)
# # with open("dynamicoutput.txt","w") as textcheck:
# for i in nuchunks:
#     # textcheck.write(str(i)+"\n")
#     pprint.pprint(i)

# HTML TEST

with open("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/nanotechnology.html", "rb") as file:

    # print(BytesIO(file.read()).read())

    upfile = UploadFile(
        file = BytesIO(file.read()),
        filename = file.name.split("/")[-1],
        size = os.path.getsize(f"C:/Users/User 3/Documents/Python scratchpad/eaglesearch/nanotechnology.html")
    )

nuchunks = txtchunker.chunk_document(upfile)
# with open("dynamicoutput.txt","w") as textcheck:
print(nuchunks)
txtchunker._ingest_text(nuchunks,upfile,"txttest", "anthony", "amarna",True)

pprint.pprint(txtchunker.search(query = 'small',txt_collection="txttest",client_id = "anthony",limit=5))



# images=[]

# for i in tqdm([1,2,3,4,5,6]):
#     with open(f"C:/Users/User 3/Documents/Python scratchpad/eaglesearch/phot/{i}.png","rb") as photo:
#         upfile = UploadFile(
#             file= BytesIO(photo.read()),
#             filename= photo.name,
#             size = os.path.getsize(f"C:/Users/User 3/Documents/Python scratchpad/eaglesearch/phot/{i}.png")
#         )
#         images.append(upfile)
# txtchunker._ingest_photos(images, "imgtest")


# print(txtchunker.search(query = 'cat',img_collection="imgtest",limit=1)[0]["doc_name"])
 
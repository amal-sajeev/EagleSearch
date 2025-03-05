from eaglesearch import EagleSearch
import pprint
from io import BytesIO
from fastapi import UploadFile
import os
from colpali_engine.models import ColQwen2, ColQwen2Processor
from tqdm import tqdm

txtchunker = EagleSearch(
    vllm_model= "",
    qdrant_api_key= os.environ["qdrantkey"],
    qdrant_url = os.environ['qdranturl'],
    MIN_CHUNK_SIZE = 600,  # Minimum Characters in each chunk
    IDEAL_CHUNK_SIZE = 1000,  # Ideal Characters in each chunk
    MAX_CHUNK_SIZE = 1200,  # Maximum Characters in each chunk
    similarity_threshold= 0.3, #Threshold for whether sentence should be added to chunk
    chunk_embedding_model = 'all-MiniLM-L6-v2', #Small Embedding model for semantic chunking.
    batch_size = 32,
    max_cache_size = 10000
)

#JSON TEST
upfile = []
# with open("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/testdata/movie-database.json", "rb") as file:

#     # print(BytesIO(file.read()).read())

#     upfilo = UploadFile(
#         file = BytesIO(file.read()),
#         filename = file.name.split("/")[-1],
#         size = os.path.getsize("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/testdata/movie-database.json")
#     )
#     upfile.append(upfilo)
    
# print(upfile.filename.split(".")[-1])
# nuchunks = txtchunker.chunk_document(upfile)

# with open("dynamicoutput.txt","w") as textcheck:
#     for i in nuchunks:
#         # textcheck.write(str(i)+"\n")
#         pprint.pprint(i)

# HTML TEST

# with open("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/testdata/nanotechnology.html", "rb") as file:

#     # print(BytesIO(file.read()).read())

#     upfilo = UploadFile(
#         file = BytesIO(file.read()),
#         filename = file.name.split("/")[-1],
#         size = os.path.getsize(f"C:/Users/User 3/Documents/Python scratchpad/eaglesearch/testdata/nanotechnology.html")
#     )
#     upfile.append(upfilo)

# nuchunks = txtchunker.chunk_document(upfile)
# with open("dynamicoutput.txt","w") as textcheck:
#     print(nuchunks)

#DOCS TEST

# with open("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/samples/sample.docx", "rb") as file:

#     # print(BytesIO(file.read()).read())

#     upfilo = UploadFile(
#         file = BytesIO(file.read()),
#         filename = file.name.split("/")[-1],
#         size = os.path.getsize(f"C:/Users/User 3/Documents/Python scratchpad/eaglesearch/samples/sample.docx")
#     )
#     upfile.append(upfilo)


#TEXT VECTOR SEARCH

# pprint.pprint(txtchunker.search(query = "Which music video had the gravity lean?",txt_collection="txttest",client_id = "anthony", bot_id = "amarna",limit=5))


#PDF TEST

with open("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/samples/sample.pdf", "rb") as file:

    # print(BytesIO(file.read()).read())

    upfilo = UploadFile(
        file = BytesIO(file.read()),
        filename = file.name.split("/")[-1],
        size = os.path.getsize(f"C:/Users/User 3/Documents/Python scratchpad/eaglesearch/samples/sample.pdf")
    )
    upfile.append(upfilo)


# images=[]

# for i in tqdm([1,2,3,4,5,6]):
#     with open(f"C:/Users/User 3/Documents/Python scratchpad/eaglesearch/phot/{i}.png","rb") as photo:
#         upfilo = UploadFile(
#             file= BytesIO(photo.read()),
#             filename= photo.name.split("/")[-1],
#             size = os.path.getsize(f"C:/Users/User 3/Documents/Python scratchpad/eaglesearch/phot/{i}.png")
#         )
#         upfile.append(upfilo)
# # txtchunker._ingest_photos(images, "imgtest")
# print(upfile)

pprint.pprint(txtchunker.ingest("anthony", "amarna",upfilo,"txttest","shodan"))

# print(txtchunker.search(query = 'cat',img_collection="imgtest",limit=1)[0]["doc_name"])
 

 #Delete test
# pprint.pprint(txtchunker.search_by_docid("636c9d4f-650d-49db-9d39-0bf9ae49532e","txttest"))

# for i in txtchunker.search_by_docid("3c105cc2-ee78-4e95-907f-ffe311008a00","qwenufo"):
#     pprint.pprint(i.id)

# print(txtchunker.delete_by_docid(" a678a6d3-a458-49b6-9a31-b5a80a4924c9","babel"))
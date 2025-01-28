from eaglesearch import EagleSearch
import pprint
from io import BytesIO

txtchunker = EagleSearch(
    max_chunk_size=1500,
    similarity_threshold= 0.5,
    qdrant_api_key= "iL-BfHumRrC_Ph8rOWLpGPT1JZu-8W6zfxxe1wm5cPaUiUeX3QSevg",
    qdrant_url ="https://c8abf992-e97b-4ccd-a2b0-046e5c9f5ee5.europe-west3-0.gcp.cloud.qdrant.io"
    )

with open("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/movie-database.json", "rb") as file:
    
    print(BytesIO(file.read()).read())

# nuchunks = txtchunker.chunk_document("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/movie-database.json")
# with open("dynamicoutput.txt","w") as textcheck:
#     for i in nuchunks:
#         textcheck.write(str(i)+"\n")

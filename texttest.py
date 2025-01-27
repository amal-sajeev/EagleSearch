from eaglesearch import EagleSearch
import pprint


txtchunker = EagleSearch(max_chunk_size=500, similarity_threshold= 0.2)

nuchunks = txtchunker.chunk_document("C:/Users/User 3/Documents/Python scratchpad/eaglesearch/Stu plan.docx")
with open("dynamicoutput.txt","w") as textcheck:
    for i in nuchunks:
        textcheck.write(str(i)+"\n")

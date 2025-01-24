from eaglesearch import EagleSearchTXT
import pprint

with open("C:/Users/User 3/Downloads/nostradamus.txt","r") as nosto:
    textcon = nosto.read()

chunks = []
curse = ["\n\n","\n","."," "]
cursenum = 0
while  len(chunks) == 0 or len(max(chunks)) > 3000:
    chunks = textcon.split(curse[cursenum])
    print(len(max(chunks)))
    if len(max(chunks)) > 500:
        cursenum+=1

with open("recurseoutput.txt","w") as textcurse:
    for i in chunks:
        textcurse.write(str(i)+"\n")

txtchunker = EagleSearchTXT(max_chunk_size=500, similarity_threshold= 0.2)
nuchunks = txtchunker.chunk_text(textcon)
with open("dynamicoutput.txt","w") as textcheck:
    for i in nuchunks:
        textcheck.write(str(i)+"\n")

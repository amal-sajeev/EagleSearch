import base64
import io
from PIL import Image

def base64_to_image(base64_string):
    """
    Convert a base64 string to an image file
    
    Parameters:
    base64_string (str): The base64 encoded image string
    
    Returns:
    PIL.Image: The decoded image
    """
    # Remove the data URL prefix if it exists
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode the base64 string
    img_data = base64.b64decode(base64_string)
    
    # Create an image object from the decoded data
    img = Image.open(io.BytesIO(img_data))
    
    return img

# Example usage:
def save_base64_image(base64_string, output_path):
    """
    Convert base64 string to image and save to file
    
    Parameters:
    base64_string (str): The base64 encoded image string
    output_path (str): Path where the image should be saved
    """
    img = base64_to_image(base64_string)
    img.save(output_path)

from eaglesearch import EagleSearch

cruncher = EagleSearch(qdrant_url ="https://c8abf992-e97b-4ccd-a2b0-046e5c9f5ee5.europe-west3-0.gcp.cloud.qdrant.io",qdrant_api_key= "iL-BfHumRrC_Ph8rOWLpGPT1JZu-8W6zfxxe1wm5cPaUiUeX3QSevg", collection_name="pdftest")

# cruncher.ingest_pdf("C:/Users/User 3/Downloads/soul_calibur.pdf", batch_size=5)

hits = cruncher.search("How do i do a quick roll?",limit=3)
n=1
payload = []
for hit in hits:
    payload.append( hit.payload["text_content"]["text_html"].split(' src="data:image/jpeg;base64,\n')[1].replace('"','').replace("\n</div>\n","") )

# with open("output.txt", "w") as out:
#     out.write(payload)
for i in payload:
    save_base64_image(i,f"{n}.png")
    n+=1
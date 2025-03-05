from fastapi import FastAPI, HTTPException, Query, File, UploadFile, status
from pydantic import BaseModel, Field
from eaglesearch import EagleSearch
import os
from colpali_engine.models import ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient, models
import torch
from typing import Union, Annotated
from io import BytesIO
from datetime import datetime

# Initiate Qwen to cache the model.
eagle = EagleSearch(
    vllm_model= ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        ).eval(),
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

app = FastAPI()

@app.post("/ingest")
def ingest(files: Union[list[UploadFile],UploadFile], client:str, bot:str, txt_collection:str, img_collection:str):

    try:
        return(eagle.ingest(
            client_id= client,
            bot_id = bot,
            files = files,
            txt_collection = txt_collection,
            img_collection= img_collection
        ))
    except Exception as e:
        print(f"ERROR UPLOADING FILE ON {datetime.now().strftime('%Y_%m_%d')} : {e}")
        return(f"ERROR: {str(e)}")

@app.post("/delete")
def delete(docid:str, collection:str):
    try:
        return(eagle.delete_by_docid(
            doc_id=docid,
            collection_name= collection
        ))
    except Exception as e:
        print(f"ERROR DELETING POINTS ON {datetime.now().strftime('%Y_%m_%d')} AT COLLECTION {collection} : {e}")
        return(f"ERRR:{str(e)}")

@app.get("/searchdoc")
def searchdoc(docid:str,collection:str):
    try:
        return(eagle.search_by_docid(
            doc_id= docid,
            collection_name= collection
        ))
    except Exception as e:
        print(f"ERROR SEARCHING POINTS ON {datetime.now().strftime('%Y_%m_%d')} AT COLLECTION {collection} : {e}")
        return(f"ERRR:{str(e)}")

@app.get("/query")
def query(query:str, imgcollection:str="", txtcollection:str="", client:str="", bot:str="", limit=10, prefetch = 100):
    try:
        return(eagle.search(
            query = query,
            limit=limit,
            prefetch_limit=prefetch,
            client_id = client,
            bot_id =bot,
            txt_collection = txtcollection,
            img_collection = imgcollection
        ))
    except Exception as e:
        print(f"ERROR QUERYING ON {datetime.now().strftime('%Y_%m_%d')} WITH QUERY [{query}] AT COLLECTIONS {imgcollection},{txtcollection} : {e}")
        return(f"ERRR:{str(e)}")

@app.get("/allcolumns")
def get_all_cols():
    try:
        return(eagle.get_collections())
    except Exception as e:
        print(f"ERROR GETTING ALL COLLECTIONS ON {datetime.now().strftime('%Y_%m_%d')}: {e}")
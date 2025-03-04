import streamlit as st
import base64
import eaglesearch
from eaglesearch import EagleSearch

eagle = EagleSearch(
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

qinput,output = st.columns([0.3,0.7])

with qinput:
    query = st.text_input("Text query for search.")
    imgcollection = eagl
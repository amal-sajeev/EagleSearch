import streamlit as st
import base64
import eaglesearch
import os
import requests
import json
import pprint

qinput,outputim, outputtx = st.columns([0.2,0.4,0.4])
results = {}
base_url = "http://localhost:8000"

def get_collections():
    return([i["name"] for i in json.loads(requests.get(f"{base_url}/allcolumns").content)["collections"]])

def query_call(query, limit, prefetch, client, bot, txtcollection, imgcollection):
    searchres = json.loads(requests.get(
        url = f"{base_url}/query?query={query}&imgcollection={imgcollection}&txtcollection={txtcollection}&client={client}&bot={bot}&limit={limit}&prefetch={prefetch}"
    ).content)

    resultfinal = {}
    pprint.pprint(searchres)
    for idx, i in enumerate(searchres):
        unique_key = f"{idx}_{hash(str(i))}"  # Create a unique key for each result
        if "metadata" in i:
            resultfinal[unique_key] = ["image", i["score"], i["page_image"]]
        elif "content" in i:
            resultfinal[unique_key] = ["text", i["score"], i["content"]["text"]]

    return resultfinal

collections = get_collections()

with qinput:
    with st.form("Search input"):
        query = st.text_input("Text query for search.")
        imgcollection = st.selectbox("Image Collection", collections )
        txtcollection = st.selectbox("Text Collection", collections )
        client = st.text_input("Client ID")
        bot = st.text_input("Bot ID")
        limit = st.number_input("Number of results to return.",value=10)
        prefetch = st.number_input("Number of patches to prefetch.",value=100)
        submitted =  st.form_submit_button("Search")
        if submitted:
            results=query_call(query,limit,prefetch,client, bot,txtcollection,imgcollection)


with outputim:
    with st.container(height=600):
        for key, value in results.items():
            if value[0] == "image":
                st.write(value[1])  # Display score
                st.write({"string":value[2]})
                st.image(eaglesearch.base64_to_image(value[2]))

with outputtx:
    with st.container(height=600):
        for key, value in results.items():
            if value[0] != "image":
                st.write(value[1])  # Display score
                st.write({"string":value[2]})
                st.write(value[2])
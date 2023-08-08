import os
import json
from flask import Flask, request
from google.cloud import storage
import numpy as np
import tempfile
import requests
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)

# googleScriptAPI = "https://script.google.com/macros/s/AKfycby3MVHQKrMBzDVeWKxy77gdvuWhXa-m-LUMnvoqLHrcHcJg53FzEeDLd-GaXLSeA8zM/exec"
googleScriptAPI = "https://script.google.com/macros/s/"


@app.route("/saveTrainedData", methods=["POST"])
def handle_webhook():
    """Handle incoming webhook from Replicate."""
    data = request.get_json()
    print("Status:", data.get("status"))
    
    webhookUrl = data.get("webhook")
    customerName, sheetApi = extract_params_from_url(webhookUrl)
    
    bucket_name = "txclassify"
    file_name = customerName + ".npy"

    embeddings_array = json_to_embeddings(data)

    save_to_gcs(bucket_name, file_name, embeddings_array)

    new_dict = {"status": "saveTrainedData", "data": None}

    new_json = json.dumps(new_dict)
    requests.post(sheetApi, data=new_json)
    
    return "", 200 


@app.route("/classify", methods=["POST"])
def handle_classify_webhook():
    """Handle incoming webhook from Replicate."""
    data = request.get_json()
    
    print("Status:", data.get("status"))
    
    webhookUrl = data.get("webhook")
    customerName, sheetApi = extract_params_from_url(webhookUrl)
    
    data_string = data.get("input").get("text_batch")

    data_list = json.loads(data_string)

    df_unclassified_data = pd.DataFrame(data_list, columns=["description"])
    print("🚀 ~ file: main.py:41 ~ df_unclassified_data:", df_unclassified_data)

    new_embeddings = json_to_embeddings(data)

    bucket_name = "txclassify" 
    file_name = customerName + ".npy"
    trained_embeddings = fetch_from_gcs(bucket_name, file_name)

    output = classify_expenses(df_unclassified_data, trained_embeddings, new_embeddings)

    df_output = pd.DataFrame.from_dict(output)
    print(df_output)

    data_dict = df_output.to_dict(orient="records")

    new_dict = {"status": "classify", "data": data_dict}

    new_json = json.dumps(new_dict)

    response = requests.post(sheetApi, data=new_json)

    return {"google_apps_script_status_code": response.status_code}, 200


def classify_expenses(df_unclassified_data, trained_embeddings, new_embeddings):
    desc_new_data = df_unclassified_data["description"]

    similarity_new_data = cosine_similarity(new_embeddings, trained_embeddings)
    similarity_df = pd.DataFrame(similarity_new_data)

    index_similarity = similarity_df.idxmax(axis=1)   

    d_output = {
        "description": desc_new_data,
        "categoryIndex": index_similarity,
    }
    return d_output
    # return None


def json_to_embeddings(json_data):
    embeddings = [item["embedding"] for item in json_data["output"]]
    embeddings_array = np.array(embeddings)

    return embeddings_array


def save_to_gcs(bucket_name, file_name, embeddings_array):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Save the numpy array as a .npy file
    with blob.open("wb") as f:
        np.save(f, embeddings_array)
        print("embedding saved")


def fetch_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the .npy file to a temporary file
    with tempfile.NamedTemporaryFile() as temp_file:
        blob.download_to_filename(temp_file.name)

        # Load the numpy array from the temporary file
        embeddings_array = np.load(temp_file.name)

    return embeddings_array

def extract_params_from_url(webhookUrl):
    
    parsed_url = urlparse(webhookUrl)
    
    query_params = parse_qs(parsed_url.query)
    print("🚀 ~ file: main.py:134 ~ query_params:", query_params)
    
    customerName = query_params.get("customerName", [None])[0]  
    print("🚀 ~ file: main.py:137 ~ customerName:", customerName)
    sheetApi = query_params.get("sheetApi", [None])[0] 
    print("🚀 ~ file: main.py:139 ~ sheetApi:", sheetApi)
    
    sheetApi = googleScriptAPI + sheetApi + "/exec"

    return customerName, sheetApi


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    # app.run(debug=True, host="0.0.0.0", port=3000)

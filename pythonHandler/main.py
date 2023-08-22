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
import os
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins

api_key = os.environ.get("replicate_API_KEY")

# googleScriptAPI = "https://script.google.com/macros/s/AKfycby3MVHQKrMBzDVeWKxy77gdvuWhXa-m-LUMnvoqLHrcHcJg53FzEeDLd-GaXLSeA8zM/exec"
googleScriptAPI = "https://script.google.com/macros/s/"


@app.route("/runTraining", methods=["POST"])
def runTraining():
    data = request.form
    file = request.files["credentialsFile"]
    file_contents = file.read().decode("utf-8")
    sheetId = data.get("spreadsheetId")
    customerName = data.get("customerName")

    range = data.get("range")

    # Save the credentials contents to a temporary file
    temp_credentials_path = "temporaryCredentials.json"
    with open(temp_credentials_path, "w") as temp_file:
        temp_file.write(file_contents)

    sheetData = getSpreadsheetData(temp_credentials_path, sheetId, range)
    df = cleanSpreadSheetData(sheetData)
    storeCleanedSheetOrder(df, customerName)
    runPrediction('saveTrainedData',customerName,'https://www.expensesorted.com/api/finishedTrainingHook',df['description'].tolist())

    print("🚀 ~ file: main.py:30 ~ df:", df)
    # response_data = {"message": "Success"}
    return {"message": "Success"}, 200


def runPrediction(apiMode, customerName, sheetApi, training_data):
    import replicate

    model = replicate.models.get("replicate/all-mpnet-base-v2")
    version = model.versions.get(
        "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
    )
    prediction = replicate.predictions.create(
        version=version,
        input={"text_batch": json.dumps(training_data)},
        webhook = f"https://pythonhandler-yxxxtrqkpa-ts.a.run.app/{apiMode}?customerName={customerName}&sheetApi={sheetApi}",
        webhook_events_filter=["completed"],
    )
    return prediction


def storeCleanedSheetOrder(df, customerName):
    bucket_name = "txclassify"
    file_name = customerName + "_index.npy"
    save_to_gcs(bucket_name, file_name, df.index.to_numpy())


def cleanSpreadSheetData(sheetData):
    df = pd.DataFrame(sheetData, columns=["index", "description"])
    df["description"] = df["description"].apply(clean_Text)
    df = df.drop_duplicates(subset=["description"])
    return df


def getSpreadsheetData(keyFile, sheetId, range):
    print("🚀 ~ file: main.py:48 ~ range:", range)
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_file(
        keyFile, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    service = build("sheets", "v4", credentials=creds)

    sheetAndRange = "Details!$" + range

    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=sheetId, range=sheetAndRange)
        .execute()
    )
    values = result.get("values", [])
    return values


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


def clean_Text(text):
    text = re.sub(
        r"[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+|\d+|\b\w{1,2}\b|xx|Value Date|Card|AUS|USA|USD|PTY|LTD|Tap and Pay|TAP AND PAY",
        "",
        str(text).strip(),
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    app.run(port=3001)
    # app.run(debug=True, host="0.0.0.0", port=3000)

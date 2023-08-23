import os
import json
from flask import Flask, request
from google.cloud import storage
from httpx import HTTPError
import numpy as np
import tempfile
import requests
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from urllib.parse import urlparse, parse_qs
import os
import re
from flask_cors import CORS
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow all origins

api_key = os.environ.get("replicate_API_KEY")

# googleScriptAPI = "https://script.google.com/macros/s/AKfycby3MVHQKrMBzDVeWKxy77gdvuWhXa-m-LUMnvoqLHrcHcJg53FzEeDLd-GaXLSeA8zM/exec"
googleScriptAPI = "https://script.google.com/macros/s/"
mainSheetId = "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"


@app.route("/runClassify", methods=["POST"])
def runClassify():
    data = request.form
    sheetId = data.get("spreadsheetId")
    customerName = data.get("customerName")

    data_range = data.get("range")

    sheetData = getSpreadsheetData(sheetId, "new_dump!$" + data_range)
    df = cleanSpreadSheetData(sheetData, ["date", "debit", "description"])

    runPrediction(
        "classify",
        customerName,
        "https://www.expensesorted.com/api/finishedTrainingHook",
        df["description"].tolist(),
    )

    print("🚀 ~ file: main.py:30 ~ df:", df)
    # response_data = {"message": "Success"}
    return {"message": "Success"}, 200


@app.route("/runTraining", methods=["POST"])
def runTraining():
    data = request.form
    sheetId = data.get("spreadsheetId")
    customerName = data.get("customerName")

    data_range = data.get("range")

    sheetData = getSpreadsheetData(sheetId, "Details!$" + data_range)
    df = cleanSpreadSheetData(
        sheetData,
        ["date", "description", "source", "debit", "credit", "category"],
    )
    df = df.drop_duplicates(subset=["description"])
    df["item_id"] = range(0, len(df))

    storeCleanedSheetOrder(df, customerName)

    res = runPrediction(        
        "saveTrainedData",
        customerName,
        "https://www.expensesorted.com/api/finishedTrainingHook",
        df["description"].tolist(),
    )
    
    print("🚀 ~ file: main.py:71 ~ res:", res)

    print("🚀 ~ file: main.py:30 ~ df:", df)

    return {"message": "Success"}, 200


# Webhook
@app.route("/saveTrainedData", methods=["POST"])
def handle_webhook():
    """Handle incoming webhook from Replicate."""
    data = request.get_json()
    print("Status:", data.get("status"))

    webhookUrl = data.get("webhook")
    customerName, sheetApi, runKey  = extract_params_from_url(webhookUrl)
    
    if checkHasRunYet(runKey,"Details!$P3:P3"):
        return "Training has already run", 200

    bucket_name = "txclassify"
    file_name = customerName + ".npy"

    embeddings_array = json_to_embeddings(data)

    save_to_gcs(bucket_name, file_name, embeddings_array)

    new_dict = {"status": "saveTrainedData", "data": None}
    
    updateRunStatus(runKey, "Details!$P3:P3")

    new_json = json.dumps(new_dict)
    requests.post(sheetApi, data=new_json)

    return "", 200


# Webhook
@app.route("/classify", methods=["POST"])
def handle_classify_webhook():
    """Handle incoming webhook from Replicate."""
    data = request.get_json()

    print("Status:", data.get("status"))

    webhookUrl = data.get("webhook")
    customerName, sheetApi, runKey = extract_params_from_url(webhookUrl)
    
    if checkHasRunYet(runKey,"Details!$P2:P2"):
        return "Classify has already run", 200

    data_string = data.get("input").get("text_batch")

    data_list = json.loads(data_string)

    df_unclassified_data = pd.DataFrame(data_list, columns=["description"])

    new_embeddings = json_to_embeddings(data)

    bucket_name = "txclassify"
    file_name = customerName + ".npy"
    trained_embeddings = fetch_from_gcs(bucket_name, file_name)

    output = classify_expenses(df_unclassified_data, trained_embeddings, new_embeddings)

    df_output = pd.DataFrame.from_dict(output)
    print("🚀 ~ file: main.py:178 ~ df_output:", df_output)

    trainedIndexCategories = fetch_from_gcs(bucket_name, customerName + "_index.npy")

    trained_categories_df = pd.DataFrame(trainedIndexCategories)

    # Merge the two DataFrames based on the index
    combined_df = df_output.merge(
        trained_categories_df, left_on="categoryIndex", right_on="item_id", how="left"
    )

    # Drop the unnecessary columns
    combined_df.drop(columns=["item_id", "categoryIndex"], inplace=True)

    newExpenses = getSpreadsheetData(mainSheetId, "new_dump!$A1:C200")
    
    df_newExpenses = prepSpreadSheetData(newExpenses, combined_df)

    try:
        append_mainSheet(df_newExpenses)
    except:
        return "error", 500
    
    updateRunStatus(runKey,"Details!$P2:P2")

    # data_dict = df_output.to_dict(orient="records")

    # new_dict = {"status": "classify", "data": data_dict}

    # new_json = json.dumps(new_dict)

    # response = requests.post(sheetApi, data=new_json)

    # return {"google_apps_script_status_code": response.status_code}, 200
    return "", 200

def generate_timestamp():
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    return timestamp

def prepSpreadSheetData(newExpenses, combined_df):
    df_newExpenses = pd.DataFrame(newExpenses, columns=["date", "debit", "description"])
    df_newExpenses["debit"] = pd.to_numeric(df_newExpenses["debit"], errors="coerce")
    df_newExpenses["credit"] = df_newExpenses["debit"].apply(
        lambda x: x if x > 0 else 0
    )
    df_newExpenses["debit"] = df_newExpenses["debit"].apply(lambda x: x if x < 0 else 0)
    df_newExpenses["category"] = combined_df["category"]
    df_newExpenses["source"] = "CBA"

    # Reorder the columns in the DataFrame
    df_newExpenses = df_newExpenses[
        ["date", "description", "source", "debit", "credit", "category"]
    ]
    return df_newExpenses
    
def updateRunStatus(new_value, data_range):
    google_service_account = os.environ.get("GOOGLE_SERVICE_ACCOUNT")
    creds = get_service_account_credentials(json.loads(google_service_account))
    service = build("sheets", "v4", credentials=creds)

    # Prepare the update request
    update_request = {
        "values": [[new_value]]
    }

    # Execute the update request
    try:
        response = service.spreadsheets().values().update(
            spreadsheetId=mainSheetId,
            range=data_range,
            valueInputOption="RAW",
            body=update_request
        ).execute()

        print("🚀 ~ Update response:", response)
        print("Cell value updated successfully.")
    except HTTPError as e:
        print(f"An error occurred: {e}")


def checkHasRunYet(runKey, data_range):
    print("🚀 ~ file: main.py:223 ~ runKey:", runKey)

    storedTimestamps = getSpreadsheetData(mainSheetId, data_range)
    if storedTimestamps:
        storedTimestamp = storedTimestamps[0][0]  # Extract the timestamp string
        print("🚀 ~ file: main.py:226 ~ storedTimestamp:", storedTimestamp)
        
        if runKey == storedTimestamp:
            return True
    return False

    
def runPrediction(apiMode, customerName, sheetApi, training_data):
    import replicate
    
    runKey = generate_timestamp()
    print("🚀 ~ file: main.py:238 ~ runKey:", runKey)

    model = replicate.models.get("replicate/all-mpnet-base-v2")
    version = model.versions.get(
        "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
    )
    
    # &runKey={runKey}
    prediction = replicate.predictions.create(
        version=version,
        input={"text_batch": json.dumps(training_data)},
        # webhook = f"https://pythonhandler-yxxxtrqkpa-ts.a.run.app/{apiMode}?customerName={customerName}&sheetApi={sheetApi}",
        webhook=f"https://555d-120-88-75-106.ngrok-free.app/{apiMode}?customerName={customerName}&runKey={runKey}&sheetApi={sheetApi}",
        webhook_events_filter=["completed"],
    )
    return prediction


def append_mainSheet(df):
    google_service_account = os.environ.get("GOOGLE_SERVICE_ACCOUNT")
    creds = get_service_account_credentials(json.loads(google_service_account))
    service = build("sheets", "v4", credentials=creds)
    # Append the new data to the end of the sheet
    append_values = df.values.tolist()

    # Prepare the append request
    append_request = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=mainSheetId,
            range="Details!$A1:G",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": append_values},
        )
    )

    # Execute the append request
    try:
        response = append_request.execute()
        print("🚀 ~ file: main.py:277 ~ response:", response)
        print("Data appended successfully.")
    except HTTPError as e:
        print(f"An error occurred: {e}")

    # Set the background color of the appended data
    # last_row_index = response.get('updates', {}).get('updatedRows', 0) - 1
    # last_row_range = f"A{last_row_index + 2}:G{last_row_index + 2}"  # Adjust the range
    # update_request = {
    #     "updateCells": {
    #         "rows": [
    #             {
    #                 "values": [
    #                     {
    #                         "userEnteredFormat": {
    #                             "backgroundColor": {"red": 0, "green": 0, "blue": 1}
    #                         }
    #                     }
    #                 ]
    #             }
    #         ],
    #         "fields": "userEnteredFormat.backgroundColor",
    #         "range": {
    #             "sheetId": mainSheetId,  # Update with the actual sheet ID
    #             "startRowIndex": last_row_index,  # Update with the actual row index
    #             "endRowIndex": last_row_index + 1,
    #             "startColumnIndex": 0,
    #             "endColumnIndex": 7,  # Assuming 7 columns (A-G)
    #         }
    #     }
    # }

    # # Execute the update request
    # batch_update_request = {"requests": [update_request]}
    # service.spreadsheets().batchUpdate(spreadsheetId=mainSheetId, body=batch_update_request).execute()


def classify_expenses(df_unclassified_data, trained_embeddings, new_embeddings):
    desc_new_data = df_unclassified_data["description"]

    similarity_new_data = cosine_similarity(new_embeddings, trained_embeddings)
    similarity_df = pd.DataFrame(similarity_new_data)

    index_similarity = similarity_df.idxmax(axis=1)
    # Which trained embedding is the new embedding most similar to?
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
        embeddings_array = np.load(temp_file.name, allow_pickle=True)

    return embeddings_array


def extract_params_from_url(webhookUrl):
    parsed_url = urlparse(webhookUrl)

    query_params = parse_qs(parsed_url.query)
    customerName = query_params.get("customerName", [None])[0]
    sheetApi = query_params.get("sheetApi", [None])[0]
    runKey = query_params.get("runKey", [None])[0]

    sheetApi = googleScriptAPI + sheetApi + "/exec"

    return customerName, sheetApi, runKey


def clean_Text(text):
    text = re.sub(
        r"[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+|\d+|\b\w{1,2}\b|xx|Value Date|Card|AUS|USA|USD|PTY|LTD|Tap and Pay|TAP AND PAY",
        "",
        str(text).strip(),
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def storeCleanedSheetOrder(df, customerName):
    bucket_name = "txclassify"
    file_name = customerName + "_index.npy"

    data_dict = {
        "item_id": df["item_id"].to_numpy(),
        "category": df["category"].to_numpy(),
        "description": df["description"].to_numpy(),
    }

    structured_array = np.array(
        list(
            zip(data_dict["item_id"], data_dict["category"], data_dict["description"])
        ),
        dtype=[("item_id", int), ("category", object), ("description", object)],
    )

    save_to_gcs(bucket_name, file_name, structured_array)


def cleanSpreadSheetData(sheetData, columns):
    df = pd.DataFrame(sheetData, columns=columns)
    df["description"] = df["description"].apply(clean_Text)
    return df


def getSpreadsheetData(sheetId, range):
    google_service_account = os.environ.get("GOOGLE_SERVICE_ACCOUNT")
    creds = get_service_account_credentials(json.loads(google_service_account))
    service = build("sheets", "v4", credentials=creds)

    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=sheetId, range=range)
        .execute()
    )
    values = result.get("values", [])
    return values


def get_service_account_credentials(json_content):
    creds = Credentials.from_service_account_info(
        json_content, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return creds


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    app.run(port=3001)
    # app.run(debug=True, host="0.0.0.0", port=3000)

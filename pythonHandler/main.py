import os
import json
from flask import Flask, request, jsonify

# from google.cloud import storage
from httpx import HTTPError
import numpy as np
import tempfile
# import requests
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from urllib.parse import urlparse, parse_qs
import re
from flask_cors import CORS
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from datetime import datetime
import replicate
from supabase import create_client, Client

url: str = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
key: str = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

BACKEND_API = os.environ.get("BACKEND_API")

app = Flask(__name__)
CORS(app)  # Allow all origins

googleScriptAPI = "https://script.google.com/macros/s/"


@app.errorhandler(500)
def handle_500(e):
    response = {"message": str(e), "type": type(e).__name__, "status_code": 500}
    return jsonify(response), 500


@app.route("/runClassify", methods=["POST"])
def runClassify():
    data = request.form
    sheetId = data.get("spreadsheetId")
    userId = data.get("userId")

    data_range = data.get("range")

    updateProcessStatus("Fetching spreadsheet data", "classify", userId)
    sheetData = getSpreadsheetData(sheetId, data_range)
    # all of these are important and could potentially include even more data.
    # i should again infer this from the type
    df = cleanSpreadSheetData(sheetData, ["date", "amount", "description"])

    updateProcessStatus("Cleaned spreadsheet data", "classify", userId)
    runPrediction(
        "classify",
        sheetId,
        userId,
        "https://www.expensesorted.com/api/finishedTrainingHook",
        df["description"].tolist(),
    )

    updateProcessStatus(
        "Fetching machine learning results - this can take a few minutes",
        "classify",
        userId,
    )
    # response_data = {"message": "Success"}
    return {"message": "Success"}, 200


@app.route("/runTraining", methods=["POST"])
def runTraining():
    data = request.form
    sheetId = data.get("spreadsheetId")
    userId = data.get("userId")

    data_range = data.get("range")

    updateProcessStatus("Fetching spreadsheet data", "training", userId)
    sheetData = getSpreadsheetData(sheetId, data_range)
    # here I need the order of the 5 different categories
    # i can infer that from the type given in the config
    # then i can also only store and use the minimum columns i.e. I only need the description column
    df = cleanSpreadSheetData(
        sheetData,
        ["source", "date", "description", "amount", "category"],
    )
    df = df.drop_duplicates(subset=["description"])
    updateProcessStatus("Cleaned up spreadsheet data", "training", userId)
    df["item_id"] = range(0, len(df))

    storeCleanedSheetOrder(df, sheetId)

    res = runPrediction(
        "training",
        sheetId,
        userId,
        "https://www.expensesorted.com/api/finishedTrainingHook",
        df["description"].tolist(),
    )

    updateProcessStatus(
        "Training machine learning model - this can take a few minutes",
        "training",
        userId,
    )

    return {"message": "Success"}, 200


# Webhook
@app.route("/training", methods=["POST"])
def handle_webhook():
    """Handle incoming webhook from Replicate."""
    data = request.get_json()
    print("Status:", data.get("status"))

    webhookUrl = data.get("webhook")
    sheetId, sheetApi, runKey, userId = extract_params_from_url(webhookUrl)

    if check_has_run_yet(runKey, "training", userId, None):
        print("Training has already run")
        return "Training has already run", 200

    updateProcessStatus(
        "Training results received, storing results", "training", userId
    )

    bucket_name = "txclassify"
    file_name = sheetId + ".npy"

    embeddings_array = json_to_embeddings(data)

    save_embeddings(bucket_name, file_name, embeddings_array)

    updateProcessStatus("completed", "training", userId)
    updateRunStatus(runKey, "training", userId)

    return "", 200


# Webhook
@app.route("/classify", methods=["POST"])
def handle_classify_webhook():
    """Handle incoming webhook from Replicate."""
    try:
        data = request.get_json()

        print("Status:", data.get("status"))

        webhookUrl = data.get("webhook")
        sheetId, sheetApi, runKey, userId = extract_params_from_url(webhookUrl)

        config = getUserConfig(userId)

        if check_has_run_yet(runKey, "categorisation", userId, config):
            print("Classify has already run")
            return "Classify has already run", 200

        updateRunStatus(runKey, "categorisation", userId)

        updateProcessStatus(
            "Categorisation results received, comparing to training data",
            "classify",
            userId,
        )

        data_string = data.get("input").get("text_batch")

        data_list = json.loads(data_string)

        df_unclassified_data = pd.DataFrame(data_list, columns=["description"])

        new_embeddings = json_to_embeddings(data)

        bucket_name = "txclassify"
        file_name = sheetId + ".npy"
        trained_embeddings = fetch_embedding(bucket_name, file_name)
        updateProcessStatus("Fetched training results", "classify", userId)
        output = classify_expenses(
            df_unclassified_data, trained_embeddings, new_embeddings
        )

        df_output = pd.DataFrame.from_dict(output)

        trainedIndexCategories = fetch_embedding(bucket_name, sheetId + "_index.npy")

        trained_categories_df = pd.DataFrame(trainedIndexCategories)

        # Merge the two DataFrames based on the index
        combined_df = df_output.merge(
            trained_categories_df,
            left_on="categoryIndex",
            right_on="item_id",
            how="left",
        )
        updateProcessStatus(
            "Compared training results, assigned categories", "classify", userId
        )
        # Drop the unnecessary columns
        combined_df.drop(columns=["item_id", "categoryIndex"], inplace=True)

        sheetRange = config["categorisationTab"] + "!" + config["categorisationRange"]

        newExpenses = getSpreadsheetData(sheetId, sheetRange)

        df_newExpenses = prepSpreadSheetData(newExpenses, combined_df)
        updateProcessStatus(
            "Preparing spreadsheet data, appending to expense sheet", "classify", userId
        )
        append_mainSheet(
            df_newExpenses, sheetId, config
        )  # Explicitly pass sheetId here
        updateProcessStatus("completed", "classify", userId)

        return "", 200

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # reset runkey if error occurs
        updateRunStatus("", "categorisation", userId)
        raise


def generate_timestamp():
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    return timestamp


def getUserConfig(userId):
    response = supabase.table("account").select("*").eq("userId", userId).execute()

    if response.data:
        config = response.data[0]  # Assuming the fetched data is a list of dictionaries
        return config

    return {}  # Return an empty dictionary if no data was found for the user


def prepSpreadSheetData(newExpenses, combined_df):
    df_newExpenses = pd.DataFrame(
        newExpenses, columns=["date", "amount", "description"]
    )
    df_newExpenses["amount"] = pd.to_numeric(df_newExpenses["amount"], errors="coerce")

    # Create a new column 'category' in df_newExpenses
    df_newExpenses = df_newExpenses.assign(
        category=np.where(
            df_newExpenses["amount"] > 0, "Credit", combined_df["category"].values
        )
    )

    df_newExpenses["source"] = "CBA"

    # Reorder the columns in the DataFrame
    df_newExpenses = df_newExpenses[
        ["source", "date", "description", "amount", "category"]
    ]
    return df_newExpenses


def updateRunStatus(new_value, mode, userId):
    if mode == "training":
        response = (
            supabase.table("account")
            .upsert({"userId": userId, "runStatusTraining": new_value})
            .execute()
        )
    else:
        response = (
            supabase.table("account")
            .upsert({"userId": userId, "runStatusCategorisation": new_value})
            .execute()
        )

    return response


def updateProcessStatus(status_text, mode, userId):
    if mode == "training":
        response = (
            supabase.table("account")
            .update({"trainingStatus": status_text})
            .eq("userId", userId)
            .execute()
        )
    else:
        response = (
            supabase.table("account")
            .update({"categorisationStatus": status_text})
            .eq("userId", userId)
            .execute()
        )

    return response


def check_has_run_yet(run_key, mode, user_id, config):
    if not config:
        config = getUserConfig(user_id)

    if mode == "training":
        stored_timestamps = config["runStatusTraining"]
    else:
        stored_timestamps = config["runStatusCategorisation"]

    if stored_timestamps:
        stored_timestamp = stored_timestamps.strip()  # Trim leading/trailing whitespace
        print("🚀 ~ file: main.py:312 ~ stored_timestamp:", stored_timestamp)
        run_key = run_key.strip()  # Trim leading/trailing whitespace
        print("🚀 ~ file: main.py:313 ~ run_key:", run_key)

        if run_key == stored_timestamp:
            print("🚀 ~ file: main.py:295 ~ stored_timestamp:", stored_timestamp)
            return True
        else:
            print("Strings are not the same")
            return False
    return False


def runPrediction(apiMode, sheetId, userId, sheetApi, training_data):
    runKey = generate_timestamp()
    print("🚀 ~ file: main.py:238 ~ runKey:", runKey)

    model = replicate.models.get("replicate/all-mpnet-base-v2")
    version = model.versions.get(
        "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
    )

    prediction = replicate.predictions.create(
        version=version,
        input={"text_batch": json.dumps(training_data)},
        webhook=f"{BACKEND_API}/{apiMode}?sheetId={sheetId}&runKey={runKey}&userId={userId}&sheetApi={sheetApi}",
        webhook_events_filter=["completed"],
    )
    return prediction


def append_mainSheet(df, sheetId, config):
    google_service_account = os.environ.get("GOOGLE_SERVICE_ACCOUNT")
    creds = get_service_account_credentials(json.loads(google_service_account))
    service = build("sheets", "v4", credentials=creds)
    # Append the new data to the end of the sheet
    append_values = df.values.tolist()

    sheetRange = config["trainingTab"] + "!" + config["trainingRange"]
    print("🚀 ~ file: main.py:352 ~ sheetRange:", sheetRange)

    # Prepare the append request
    append_request = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=sheetId,
            range=sheetRange,
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": append_values},
        )
    )

    # Execute the append request
    try:
        append_request.execute()
        print("Data appended successfully.")
    except HTTPError as e:
        print(f"An error occurred: {e}")


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


def save_embeddings(bucket_name, file_name, embeddings_array):
    # save_to_gcs(bucket_name, file_name, embeddings_array)
    save_to_supabase(bucket_name, file_name, embeddings_array)


def fetch_embedding(bucket_name, file_name):
    # fetch_from_gcs(bucket_name, file_name)
    return fetch_from_supabase(bucket_name, file_name)


def fetch_from_supabase(bucket_name, file_name):
    # Download the .npy file to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
        response = supabase.storage.from_(bucket_name).download(file_name)

        # Write the content to the temporary file
        temp_file.write(response)
        temp_file.flush()

    # Load the numpy array from the temporary file
    embeddings_array = np.load(temp_file.name, allow_pickle=True)['arr_0']
    # print("🚀 ~ file: main.py:416 ~ embeddings_array:", embeddings_array)

    # Delete the temporary file
    os.unlink(temp_file.name)

    return embeddings_array


# def save_to_gcs(bucket_name, file_name, embeddings_array):
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(file_name)

#     # Save the numpy array as a .npy file
#     with blob.open("wb") as f:
#         np.save(f, embeddings_array)
#         print("embedding saved")


def save_to_supabase(bucket_name, file_name, embeddings_array):
    # Create a temporary file to store the NumPy array
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
        # Save the NumPy array to the temporary file
        np.savez_compressed(temp_file, embeddings_array)

    # Get the file path of the temporary file
    file_path = temp_file.name

    # Upload the temporary file to Supabase Storage
    with open(file_path, "rb") as f:
        response = supabase.storage.from_(bucket_name).upload(
            file_name, f, file_options={"x-upsert": "true"}
        )

    # Remove the temporary file
    os.unlink(file_path)

    if response.status_code == 200:
        # Successful upload
        print("Upload successful")
    else:
        # Handle the error
        print(f"Upload failed with status code {response.status_code}")
        # Optionally, you can print the response content for more details:
    print(response.content)


# def fetch_from_gcs(bucket_name, file_name):
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(file_name)

#     # Download the .npy file to a temporary file
#     with tempfile.NamedTemporaryFile() as temp_file:
#         blob.download_to_filename(temp_file.name)

#         # Load the numpy array from the temporary file
#         embeddings_array = np.load(temp_file.name, allow_pickle=True)

#     return embeddings_array


def extract_params_from_url(webhookUrl):
    parsed_url = urlparse(webhookUrl)

    query_params = parse_qs(parsed_url.query)
    sheetId = query_params.get("sheetId", [None])[0]
    sheetApi = query_params.get("sheetApi", [None])[0]
    runKey = query_params.get("runKey", [None])[0]
    userId = query_params.get("userId", [None])[0]

    sheetApi = googleScriptAPI + sheetApi + "/exec"

    return sheetId, sheetApi, runKey, userId


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

    save_embeddings(bucket_name, file_name, structured_array)


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
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    # app.run(port=3001)

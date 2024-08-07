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
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials
from datetime import datetime
import replicate
from supabase import create_client, Client
from urllib.parse import urlparse

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
    data = request.get_json()
    print("🚀 ~ file: main.py:43 ~ data:", data)
    sheetId = data.get("expenseSheetId")
    userId = data.get("userId")
    columnOrderCategorisation = data.get("columnOrderCategorisation")
    categorisationTab = data.get("categorisationTab")

    # data_range = data.get("range")
    firstColumn, secondColumn, columnHeader = get_column_names(
        columnOrderCategorisation
    )

    updateProcessStatus("Fetching spreadsheet data", "classify", userId)
    
    try:
        sheetData = getSpreadsheetData(
            sheetId, categorisationTab + "!" + firstColumn + ":" + secondColumn
        )
    except Exception as e:
        updateProcessStatus(f"Error fetching spreadsheet data: please check access via technical user", "training", userId)
        raise

    df = cleanSpreadSheetData(sheetData, columnHeader)

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
    data = request.get_json()
    sheetId = data.get("expenseSheetId")
    userId = data.get("userId")
    columnOrderTraining = data.get("columnOrderTraining")
    trainingTab = data.get("trainingTab")    

    descriptionColumn, categoryColumn, columnOrder = get_column_names_and_types(        
        columnOrderTraining
    )
    
    if not all([trainingTab, categoryColumn, descriptionColumn]):
        
        return {"message": "Missing required data"}, 400

    updateProcessStatus("Fetching spreadsheet data", "training", userId)
    try:
        sheetData = getSpreadsheetData(
            sheetId, trainingTab + "!" + categoryColumn + ":" + descriptionColumn
        )
    except Exception as e:
        updateProcessStatus(f"Error fetching spreadsheet data: please check access via technical user", "training", userId)
        raise

    df = cleanSpreadSheetData(
        sheetData[1:],
        # ["source", "date", "description", "amount", "category"],
        columnOrder,
    )
    df = df.drop_duplicates(subset=["description"])
    updateProcessStatus("Cleaned up spreadsheet data", "training", userId)
    df["item_id"] = range(0, len(df))

    storeCleanedSheetOrder(df, sheetId)

    sheetApi = "https://www.expensesorted.com/api/finishedTrainingHook"
    
    print(f"About to call runPrediction with sheetApi URL: {sheetApi}")
    res = runPrediction(
        "training",
        sheetId,
        userId,
        sheetApi,
        df["description"].tolist(),
    )
    print(f"runPrediction result: {res}")

    updateProcessStatus(
        "Training machine learning model - this can take a few minutes",
        "training",
        userId,
    )

    return {"message": "Success"}, 200

def is_valid_https_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme == 'https', result.netloc, result.path])
    except:
        return False

# Webhook
@app.route("/training", methods=["POST"])
def handle_webhook():
    """Handle incoming webhook from Replicate."""
    try:
        data = request.get_json()
        # print("Received data:", data)  # Log the received data

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

        try:
            embeddings_array = json_to_embeddings(data)
        except KeyError as e:
            print(f"KeyError in json_to_embeddings: {e}")
            print("Data structure:", data)
            return "Invalid data structure", 400

        save_embeddings(bucket_name, file_name, embeddings_array)

        updateProcessStatus("completed", "training", userId)
        updateRunStatus(runKey, "training", userId)

        return "", 200
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return str(e), 500


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
        print("🚀 ~ file: main.py:187 ~ df_unclassified_data:", df_unclassified_data)

        new_embeddings = json_to_embeddings(data)

        bucket_name = "txclassify"
        file_name = sheetId + ".npy"
        trained_embeddings = fetch_embedding(bucket_name, file_name)

        updateProcessStatus("Fetched training results", "classify", userId)
        output = classify_expenses(
            df_unclassified_data, trained_embeddings, new_embeddings
        )

        # for some reason the df_output here is shorter?!
        df_output = pd.DataFrame.from_dict(output)
        print("🚀 ~ file: main.py:203 ~ df_output:", df_output)

        trainedIndexCategories = fetch_embedding(bucket_name, sheetId + "_index.npy")
        print("🚀 ~ file: main.py:190 ~ trainedIndexCategories:", trainedIndexCategories)

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
        print("🚀 ~ file: main.py:203 ~ combined_df:", combined_df)

        print("🚀 ~ file: main.py:220 ~ :", config)
        firstColumn, secondColumn, sourceColumns = get_column_names(
            config["columnOrderCategorisation"]
        )
        
        _, _, targetColumns = get_column_names(
            config["columnOrderTraining"]
        )

        sheetRange = (
            config["categorisationTab"] + "!" + firstColumn + ":" + secondColumn
        )
        print("🚀 ~ file: main.py:227 ~ sheetRange:", sheetRange)
        
        try:
            newExpenses = getSpreadsheetData(sheetId, sheetRange)        
        except Exception as e:
            updateProcessStatus(f"Error fetching spreadsheet data: please check access via technical user", "training", userId)
            raise   

        df_newExpenses = prepSpreadSheetData(newExpenses, combined_df, sourceColumns, targetColumns)
        updateProcessStatus(
            "Preparing spreadsheet data, appending to expense sheet", "classify", userId
        )
        append_mainSheet(df_newExpenses, sheetId, config)
        updateProcessStatus("completed", "classify", userId)

        return "", 200

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # reset runkey if error occurs
        updateRunStatus("", "categorisation", userId)
        raise


def get_column_names_and_types(columnOrderTraining):
    # Sort the list by 'index'
    sorted_list = sorted(columnOrderTraining, key=lambda x: x["index"])

    # Initialize variables
    description_name = None
    category_name = None
    column_order = []
    start_collecting = False

    # Iterate over the sorted list
    for item in sorted_list:
        if item["type"] == "description":
            description_name = item["name"]
            start_collecting = True

        if start_collecting:
            column_order.append(item["type"])

        if item["type"] == "category":
            category_name = item["name"]
            break

    return description_name, category_name, column_order


def get_column_names(columnOrder):
    # Sort the list by 'index'
    if columnOrder is None:
        # Handle the case when columnOrder is None
        # You might want to return some default value or raise an error
        return None, None, None
    sorted_list = sorted(columnOrder, key=lambda x: x["index"])

    # Get the names of the first and last items
    first_name = sorted_list[0]["name"]
    last_name = sorted_list[-1]["name"]

    # Get all types in an array
    types = [item["type"] for item in sorted_list]

    return first_name, last_name, types


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


def prepSpreadSheetData(newExpenses, df_newCategories, sourceColumns, targetColumns):
    df_newExpenses = pd.DataFrame(newExpenses, columns=sourceColumns)
    
    if "amount" in df_newExpenses.columns:
        df_newExpenses["amount"] = pd.to_numeric(
            df_newExpenses["amount"], errors="coerce"
        )

        # Create a new column 'category' in df_newExpenses
        df_newExpenses = df_newExpenses.assign(
            category=np.where(
                df_newExpenses["amount"] > 0, "Credit", df_newCategories["category"].values
            )
        )
    else:
        df_newExpenses = df_newExpenses.assign(category=df_newCategories["category"].values)

    # Check if 'source' exists in both sourceColumns and targetColumns
    if "source" in sourceColumns and "source" in targetColumns:
        pass  # 'source' will be transferred automatically when reordering columns
    elif "source" in targetColumns:  # 'source' only exists in targetColumns
        df_newExpenses["source"] = "Bank 1"  # fill with default value

    # Reorder the columns in the DataFrame based on the target categorised expenses sheet
    df_newExpenses = df_newExpenses[targetColumns]
    
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

    try:
        # Validate sheetApi URL
        if not is_valid_https_url(sheetApi):
            raise ValueError(f"Invalid sheetApi URL: {sheetApi}")

        model = replicate.models.get("replicate/all-mpnet-base-v2")
        version = model.versions.get(
            "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305"
        )

        webhook = f"{BACKEND_API}/{apiMode}?sheetId={sheetId}&runKey={runKey}&userId={userId}&sheetApi={sheetApi}"
        
        prediction = replicate.predictions.create(
            version=version,
            input={"text_batch": json.dumps(training_data)},
            webhook=webhook,
            webhook_events_filter=["completed"],
        )
        return prediction
    except Exception as e:
        print(f"Error in runPrediction: {str(e)}")
        raise


def append_mainSheet(df, sheetId, config):
    google_service_account = os.environ.get("GOOGLE_SERVICE_ACCOUNT")
    creds = get_service_account_credentials(json.loads(google_service_account))
    service = build("sheets", "v4", credentials=creds)
    # Append the new data to the end of the sheet
    append_values = df.values.tolist()
    
    firstColumn, lastColumn, _ = get_column_names(
        config["columnOrderTraining"]
    )

    sheetRange = config["trainingTab"] + "!" + firstColumn + ":" + lastColumn

    # Prepare the append request
    append_request = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=sheetId,
            range=sheetRange,
            valueInputOption="USER_ENTERED",
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
    print("🚀 ~ file: main.py:451 ~ desc_new_data:", desc_new_data)

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
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
        response = supabase.storage.from_(bucket_name).download(file_name)

        # Write the content to the temporary file
        temp_file.write(response)
        temp_file.flush()

    # Load the numpy array from the temporary file
    embeddings_array = np.load(temp_file.name, allow_pickle=True)["arr_0"]
    # print("🚀 ~ file: main.py:416 ~ embeddings_array:", embeddings_array)

    # Delete the temporary file
    os.unlink(temp_file.name)

    return embeddings_array


def save_to_supabase(bucket_name, file_name, embeddings_array):
    # Create a temporary file to store the NumPy array
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
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
    try:
        google_service_account = os.environ.get("GOOGLE_SERVICE_ACCOUNT")
        if not google_service_account:
            raise ValueError("Environment variable 'GOOGLE_SERVICE_ACCOUNT' is not set")

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

    except HttpError as he:
        if he.resp.status == 403:
            print("Access to the Google Sheet was denied.")
        else:
            print(f"HTTP Error while accessing the Google Sheets API: {he}")
        raise
    except ValueError as ve:
        print(f"Value Error: {ve}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def get_service_account_credentials(json_content):
    creds = Credentials.from_service_account_info(
        json_content, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return creds


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    app.run(port=3001)

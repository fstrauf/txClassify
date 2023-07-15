# import datetime
from io import StringIO
import json
# import logging
from flask import Flask, request, make_response
import nltk
from nltk.tokenize import word_tokenize
import re

# import csv
# import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# from faunadb import query as q
# from faunadb.client import FaunaClient
# from faunadb.objects import Ref
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

url: str = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
key: str = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)

# logging.basicConfig(level=logging.DEBUG)


@app.route("/api/classify", methods=["POST"])
def classify():
    file = request.files["file"]
    app.logger.debug(f"Received file: {file}")
    file_contents = file.read().decode("utf-8")
    csv_data = StringIO(file_contents)

    df_unclassified_data = pd.read_csv(
        csv_data, header=None, names=["Date", "Amount", "Narrative", "old"]
    )

    df_unclassified_data.drop("old", axis=1, inplace=True)
    # print(df_unclassified_data)

    trained_embeddings_BERT = np.load("trained_embeddings.npy")
    classified_file_path = "all_expenses_classified.csv"
    df_classified_data = pd.read_csv(classified_file_path)
    print("Found trained data")

    output = classify_expenses(
        df_unclassified_data, trained_embeddings_BERT, df_classified_data
    )

    df_output = pd.DataFrame.from_dict(output)

    # Convert the DataFrame to JSON
    json_data = df_output.to_json(orient="records")

    response = make_response(json_data)
    response.headers["Content-Type"] = "application/json"

    # Return the JSON data
    return json_data


@app.route("/api/adminUploadFauna", methods=["POST"])
def adminUploadFauna():
    file = request.files["file"]
    file_contents = file.read().decode("utf-8")
    csv_data = StringIO(file_contents)

    all_classified_expenses = pd.read_csv(
        csv_data,
        header=None,
        names=["Date", "Narrative", "Debit Amount", "Credit Amount", "Categories"],
    )

    json_expenses = all_classified_expenses.to_json(orient="records")
    expenses_list = json.loads(json_expenses)

    result = client.query(
        q.map_(
            q.lambda_(
                "data",
                q.create(
                    q.collection("Expenses"),
                    {"data": q.var("data")},
                ),
            ),
            expenses_list,
        )
    )
    return result


@app.route("/api/adminUpload", methods=["POST"])
def adminUpload():
    file = request.files["file"]
    file_contents = file.read().decode("utf-8")
    csv_data = StringIO(file_contents)

    all_classified_expenses = pd.read_csv(
        csv_data,
        header=None,
        names=["date", "description", "debitamount", "creditamount", "category"],
    )
    
    all_classified_expenses["creditamount"] = all_classified_expenses["creditamount"].fillna(0)

    json_expenses = all_classified_expenses.to_json(orient="records")
    expenses_list = json.loads(json_expenses)

    data, count = supabase.table('expenses').insert(expenses_list).execute()

    response = make_response({"message": "Success: Retraining completed successfully!"})
    response.headers["Content-Type"] = "application/json"
    return response

@app.route("/api/convertToCSV", methods=["POST"])
def convertToCSV():
    data = request.json
    df = pd.DataFrame(data)

    csv_data = df.to_csv(index=False)
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=exported_data.csv"
    response.headers["Content-type"] = "text/csv"

    return response


@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        data = request.json
        df_reclassified = pd.DataFrame(data)

        df_reclassified = df_reclassified.rename(columns={"Amount": "Debit Amount"})

        classified_file_path = "all_expenses_classified.csv"
        df_classified_data = pd.read_csv(classified_file_path)

        df_combined = pd.concat(
            [df_classified_data, df_reclassified], ignore_index=True
        )
        df_combined = df_combined.drop_duplicates()

        text_descriptions = df_combined["Narrative"]
        text_BERT = text_descriptions.apply(lambda x: clean_text_BERT(x))

        bert_input = text_BERT.tolist()
        model = SentenceTransformer("paraphrase-mpnet-base-v2")
        embeddings = model.encode(bert_input, show_progress_bar=True)
        embedding_BERT = np.array(embeddings)

        np.save("trained_embeddings.npy", embedding_BERT)
        df_combined.to_csv("all_expenses_classified.csv", index=False)

        response = make_response(
            {"message": "Success: Retraining completed successfully!"}
        )
        response.headers["Content-Type"] = "application/json"
        return response

    except Exception as e:
        response = make_response({"error": str(e)})
        response = make_response({"error": ""})
        response.headers["Content-Type"] = "application/json"
        return response

def clean_text_BERT(text):
    text = text.lower()

    text = re.sub(
        r"[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+|\d+",
        "",
        str(text).strip(),
    )

    text_list = word_tokenize(text)
    result = " ".join(text_list)
    return result


def classify_expenses(df_unclassified_data, trained_embeddings_BERT, df_training_data):
    desc_new_data = df_unclassified_data["Narrative"]
    amount = df_unclassified_data["Amount"]
    date = df_unclassified_data["Date"]

    text_BERT = desc_new_data.apply(lambda x: clean_text_BERT(x))

    bert_input = text_BERT.tolist()
    model = SentenceTransformer("paraphrase-mpnet-base-v2")
    embeddings_new = model.encode(bert_input, show_progress_bar=True)
    embedding_BERT_new = np.array(embeddings_new)

    similarity_new_data = cosine_similarity(embedding_BERT_new, trained_embeddings_BERT)
    similarity_df = pd.DataFrame(similarity_new_data)

    index_similarity = similarity_df.idxmax(axis=1)

    data_inspect = df_training_data.iloc[index_similarity, :].reset_index(drop=True)

    annotation = data_inspect["Categories"]

    d_output = {
        "Date": date,
        "Amount": amount,
        "Narrative": desc_new_data,
        "Categories": annotation,
    }
    return d_output


def run_test_classify(test_file_name):
    with app.test_client() as client:
        with open(test_file_name, "rb") as file:
            response = client.post("/api/classify", data={"file": file})
            # print(response.get_json())


def run_test_adminUpload(test_file_name):
    with app.test_client() as client:
        with open(test_file_name, "rb") as file:
            response = client.post("/api/adminUpload", data={"file": file})
            # print(response.get_json())


def run_test_retrain(test_data):
    with app.test_client() as client:
        response = client.post("/api/retrain", json=test_data)
        print(response.get_json())


if __name__ == "__main__":
    # app.run()
    test_file_name = "all_expenses_classified.csv"
    run_test_adminUpload(test_file_name)

    # test_data = [
    #     {
    #         "Date": "27/05/2023",
    #         "Amount": -27.88,
    #         "Narrative": "CRUISIN MOTORHOMES CAMBRIDGE AUS Card xx6552 Value Date: 25/05/2023",
    #         "Categories": "Travel",
    #     },
    #     {
    #         "Date": "26/05/2023",
    #         "Amount": -5.44,
    #         "Narrative": "TRANSPORTFORNSW TAP SYDNEY AUS Card xx0033 Value Date: 23/05/2023",
    #         "Categories": "Transport",
    #     },
    #     {
    #         "Date": "24/05/2023",
    #         "Amount": -7.58,
    #         "Narrative": "Girdlers Dee Why Dee Why NS AUS Card xx6552 Value Date: 22/05/2023",
    #         "Categories": "DinnerBars",
    #     },
    # ]
    # run_test_retrain(test_data)

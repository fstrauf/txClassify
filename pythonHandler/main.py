import os
import json
from flask import Flask, request
from google.cloud import storage
import numpy as np
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)


@app.route("/saveTrainedData", methods=["POST"])
def handle_webhook():
    """Handle incoming webhook from Replicate."""
    data = request.get_json()
    print("Status:", data.get("status"))

    bucket_name = "txclassify"  # Replace with your bucket name
    file_name = "trained_embeddings.npy"

    embeddings_array = json_to_embeddings(data)

    save_to_gcs(bucket_name, file_name, embeddings_array)
    return "", 200  # Respond with a 200 status code


@app.route("/classify", methods=["POST"])
def handle_classify_webhook():
    """Handle incoming webhook from Replicate."""
    data = request.get_json()
    print("Status:", data.get("status"))
    
    data_string = data.get("input").get("text_batch")

    # Convert the string representation of the list into an actual list
    data_list = json.loads(data_string)

    # Convert the list into a DataFrame
    df_unclassified_data = pd.DataFrame(data_list, columns=["description"])
    print("🚀 ~ file: main.py:41 ~ df_unclassified_data:", df_unclassified_data)

    new_embeddings = json_to_embeddings(data)

    ## turn input to df_unclassified

    bucket_name = "txclassify"  # Replace with your bucket name
    file_name = "trained_embeddings.npy"
    trained_embeddings = fetch_from_gcs(bucket_name, file_name)

    output = classify_expenses(df_unclassified_data, trained_embeddings, new_embeddings)

    df_output = pd.DataFrame.from_dict(output)
    print(df_output)

    # Convert the DataFrame to JSON
    json_data = df_output.to_json(orient="records")

    return "", 200  # Respond with a 200 status code


def classify_expenses(df_unclassified_data, trained_embeddings, new_embeddings):
    desc_new_data = df_unclassified_data["description"]

    similarity_new_data = cosine_similarity(new_embeddings, trained_embeddings)
    similarity_df = pd.DataFrame(similarity_new_data)

    index_similarity = similarity_df.idxmax(axis=1)
    # print("🚀 ~ file: main.py:70 ~ index_similarity:", index_similarity)

    # i need to make sure that the indices match

    # annotation = data_inspect["category"]

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


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    app.run(debug=True, host="0.0.0.0", port=3000)

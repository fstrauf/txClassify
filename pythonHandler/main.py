import os
import json
from flask import Flask, request
from google.cloud import storage
import numpy as np

app = Flask(__name__)

@app.route("/", methods=['POST'])
def handle_webhook():
    """Handle incoming webhook from Replicate."""
    # print(request.data)  # Print the raw request data
    data = request.get_json()
    print("Status:", data.get('status'))
    
    bucket_name = "txclassify"  # Replace with your bucket name
    file_name = "trained_embeddings.npy"
    save_to_gcs(bucket_name, file_name, data)
    # print("data saved")
    return '', 200  # Respond with a 200 status code


def save_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Extract the embeddings from the data and convert to a numpy array
    embeddings = [item['embedding'] for item in data['output']]
    embeddings_array = np.array(embeddings)

    # Save the numpy array as a .npy file
    with blob.open("wb") as f:
        np.save(f, embeddings_array)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    # app.run(debug=True, host="0.0.0.0", port=3000)

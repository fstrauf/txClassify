import functions_framework
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from google.cloud import storage
import os
import nltk
nltk.download('punkt')

# os.environ["TOKENIZERS_PARALLELISM"] = "false"


@functions_framework.http
def txclassify(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    mode = None
    csv_file = None
    if request_json and 'mode' in request_json:
        mode = request_json['mode']
    elif request_args and 'mode' in request_args:
        mode = request_args['mode']

    if 'file' in request.files:
        csv_file = request.files['file']

    if mode == 'train':
        # Handle training logic
        if csv_file:
            # Read the CSV file
            df = pd.read_csv(csv_file, names=["date", "description", "amountdebit", "amountcredit", "category"])
            df_combined = df
            text_descriptions = df_combined["description"]
            text_BERT = text_descriptions.apply(lambda x: clean_text_BERT(x))
            
            bert_input = text_BERT.tolist()
            model = SentenceTransformer("paraphrase-mpnet-base-v2")
            embeddings = model.encode(bert_input, show_progress_bar=True)
            embedding_BERT = np.array(embeddings)
            # np.save("trained_embeddings.npy", embedding_BERT)
            
            # Save the embeddings to Google Cloud Storage
            bucket_name = "txclassify"  # Replace with your bucket name
            file_name = "trained_embeddings.npy"
            save_to_gcs(bucket_name, file_name, embedding_BERT)

            return 'Training completed!'
        else:
            return 'No CSV file provided for training.'

    elif mode == 'classify':
        # Handle classification logic
        if csv_file:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            # Perform classification using the DataFrame (df)
            # ...
            return 'Classification completed!'
        else:
            return 'No CSV file provided for classification.'

    else:
        return 'Invalid mode. Please provide either "train" or "classify."'
    
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

def save_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Save the data as a numpy .npy file
    with blob.open("wb") as f:
        np.save(f, data)

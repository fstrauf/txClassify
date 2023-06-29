from io import StringIO
import logging
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
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)

logging.basicConfig(level=logging.DEBUG)

@app.route("/api/classify", methods=["POST"])
def classify():
    file = request.files["file"]
    app.logger.debug(f"Received file: {file}")
    file_contents = file.read().decode("utf-8")
    csv_data = StringIO(file_contents)

    df_unclassified_data = pd.read_csv(csv_data, header=None, names=['Date', 'Amount', 'Narrative', 'old'])

    df_unclassified_data.drop('old', axis=1, inplace=True)
    print(df_unclassified_data)
    
    trained_embeddings_BERT = np.load("trained_embeddings.npy")
    classified_file_path = "all_expenses_classified.csv"
    df_classified_data = pd.read_csv(classified_file_path)
    print("Found trained data")
    
    output = classify_expenses(df_unclassified_data, trained_embeddings_BERT, df_classified_data)
    
    df_output = pd.DataFrame.from_dict(output)
    
    # Convert the DataFrame to JSON
    json_data = df_output.to_json(orient="records")
    
    response = make_response(json_data)    
    response.headers["Content-Type"] = "application/json"
    
    # Return the JSON data
    return json_data



@app.route("/api/user")
def hello_user():
    return "<p>Hello, user!</p>"


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
    desc_new_data = df_unclassified_data['Narrative']
    amount = df_unclassified_data['Amount']
    date = df_unclassified_data['Date']
    
    text_BERT = desc_new_data.apply(lambda x: clean_text_BERT(x))

    bert_input = text_BERT.tolist()
    model = SentenceTransformer('paraphrase-mpnet-base-v2') 
    embeddings_new = model.encode(bert_input, show_progress_bar = True)
    embedding_BERT_new = np.array(embeddings_new)

    similarity_new_data = cosine_similarity(embedding_BERT_new, trained_embeddings_BERT)
    similarity_df = pd.DataFrame(similarity_new_data)

    index_similarity = similarity_df.idxmax(axis = 1)

    data_inspect = df_training_data.iloc[index_similarity, :].reset_index(drop = True)

    annotation = data_inspect['Categories']

    d_output = {
                'Date': date,
                'Amount': amount,
                'Narrative': desc_new_data,
                'Categories': annotation,
                }
    return d_output

def run_test(test_file_name):
    with app.test_client() as client:
        with open(test_file_name, "rb") as file:
            response = client.post("/api/classify", data={"file": file})
            # print(response.get_json())

if __name__ == "__main__":
    # test_file_name = "new_expenses.csv"
    # run_test(test_file_name)
    app.run()
    
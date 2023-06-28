from flask import Flask, request, make_response
from nltk.tokenize import word_tokenize
import re
import csv
import json
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)

@app.route("/api/classify", methods=["POST"])
def classify():
    file = request.files["file"]
    
    file_contents = file.read().decode("utf-8")
    csv_data = list(csv.DictReader(file_contents.splitlines()))
    
    # Convert the CSV data to JSON
    json_data = json.dumps(csv_data)
    
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

if __name__ == "__main__":
    app.run()
from flask import Flask, request
from nltk.tokenize import word_tokenize
import re
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)

@app.route("/api/classify", methods=["POST"])
def classify():
    file = request.files["file"]
    print(file)
    # Perform classification or processing on the uploaded file here
    
    # Example: Read the file contents as a string
    file_contents = file.read().decode("utf-8")
    print(file_contents)
    # Return the file contents as a string
    return file_contents

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
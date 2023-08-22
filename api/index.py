
# from io import StringIO
# import io
# import json
# from flask import Flask, request, make_response
# # import nltk
# from nltk.tokenize import word_tokenize
# import re
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import os
# from supabase import create_client, Client
# from dotenv import load_dotenv
# import tempfile

# # Load environment variables from .env file
# load_dotenv()

# # url: str = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
# # key: str = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
# supabase: Client = create_client(url, key)

# app = Flask(__name__)

# @app.route("/api/classify", methods=["POST"])
# def classify():
#     file = request.files["file"]
#     app.logger.debug(f"Received file: {file}")
#     file_contents = file.read().decode("utf-8")
#     csv_data = StringIO(file_contents)

#     df_unclassified_data = pd.read_csv(
#         csv_data, header=None, names=["Date", "Amount", "Narrative", "old"]
#     )

#     df_unclassified_data.drop("old", axis=1, inplace=True)
#     # print(df_unclassified_data)

#     trained_embeddings_BERT = np.load("trained_embeddings.npy")
    
#     # response = supabase.table('expenses').select("*").execute()
#     # df_classified_data = pd.DataFrame(response.data)
#     # classified_file_path = "all_expenses_classified.csv"
#     # df_classified_data = pd.read_csv(classified_file_path)
#     # print(df_classified_data)
    
#     print("Found trained data")

#     output = classify_expenses(
#         # df_unclassified_data, trained_embeddings_BERT, df_classified_data
#         df_unclassified_data, trained_embeddings_BERT
#     )

#     df_output = pd.DataFrame.from_dict(output)
#     print(df_output)

#     # Convert the DataFrame to JSON
#     json_data = df_output.to_json(orient="records")

#     response = make_response(json_data)
#     response.headers["Content-Type"] = "application/json"

#     # Return the JSON data
#     return json_data

# @app.route("/api/adminUpload", methods=["POST"])
# def adminUpload():
#     file = request.files["file"]
#     file_contents = file.read().decode("utf-8")
#     csv_data = StringIO(file_contents)

#     all_classified_expenses = pd.read_csv(
#         csv_data,
#         header=None,
#         names=["date", "description", "debitamount", "creditamount", "category"],
#     )
    
#     all_classified_expenses["creditamount"] = all_classified_expenses["creditamount"].fillna(0)

#     json_expenses = all_classified_expenses.to_json(orient="records")
#     expenses_list = json.loads(json_expenses)

#     data, count = supabase.table('expenses').insert(expenses_list).execute()

#     response = make_response({"message": "Success: Retraining completed successfully!"})
#     response.headers["Content-Type"] = "application/json"
#     return response

# @app.route("/api/convertToCSV", methods=["POST"])
# def convertToCSV():
#     data = request.json
#     df = pd.DataFrame(data)

#     csv_data = df.to_csv(index=False)
#     response = make_response(csv_data)
#     response.headers["Content-Disposition"] = "attachment; filename=exported_data.csv"
#     response.headers["Content-type"] = "text/csv"

#     return response


# @app.route("/api/retrain", methods=["POST"])
# def retrain():
#     try:
#         new_reclassified = request.json
#         df_reclassified = pd.DataFrame(new_reclassified)

#         # df_reclassified = df_reclassified.rename(columns={"Amount": "debitamount"})

#         # classified_file_path = "all_expenses_classified.csv"        
#         # df_classified_data = pd.read_csv(classified_file_path)
#         response = supabase.table('expenses').select("*").execute()
#         df_classified_data = pd.DataFrame(response.data)
#         # print("🚀 ~ file: index.py:123 ~ df_classified_data:", df_classified_data)
        

#         df_combined = pd.concat(
#             [df_classified_data, df_reclassified], ignore_index=True
#         )
#         df_combined = df_combined.drop_duplicates()
#         # print("🚀 ~ file: index.py:130 ~ df_combined:", df_combined)

#         text_descriptions = df_combined["description"]
#         #drop text duplicates here too
#         text_BERT = text_descriptions.apply(lambda x: clean_text_BERT(x))

#         bert_input = text_BERT.tolist()
#         model = SentenceTransformer("paraphrase-mpnet-base-v2")
#         embeddings = model.encode(bert_input, show_progress_bar=True)
#         embedding_BERT = np.array(embeddings)
        
#         with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_file:
#             np.save(temp_file, embedding_BERT)

#         response = supabase.storage.from_('BERT_DATA').upload("embedding_BERT.npy", temp_file.name)
#         print("🚀 ~ file: index.py:137 ~ response:", response)

#         data, count = supabase.table('expenses').insert(new_reclassified).execute()
#         # np.save("trained_embeddings.npy", embedding_BERT)
#         # df_combined.to_csv("all_expenses_classified.csv", index=False)

#         response = make_response(
#             {"message": "Success: Retraining completed successfully!"}
#         )
#         response.headers["Content-Type"] = "application/json"
#         return response

#     except Exception as e:
#         response = make_response({"error": str(e)})
#         response.headers["Content-Type"] = "application/json"
#         return response

# def clean_text_BERT(text):
#     text = text.lower()

#     text = re.sub(
#         r"[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+|\d+",
#         "",
#         str(text).strip(),
#     )

#     text_list = word_tokenize(text)
#     result = " ".join(text_list)
#     return result


# def classify_expenses(df_unclassified_data, trained_embeddings_BERT):
#     desc_new_data = df_unclassified_data["Narrative"]
#     amount = df_unclassified_data["Amount"]
#     date = df_unclassified_data["Date"]

#     text_BERT = desc_new_data.apply(lambda x: clean_text_BERT(x))

#     bert_input = text_BERT.tolist()
#     model = SentenceTransformer("paraphrase-mpnet-base-v2")
#     embeddings_new = model.encode(bert_input, show_progress_bar=True)
#     embedding_BERT_new = np.array(embeddings_new)

#     similarity_new_data = cosine_similarity(embedding_BERT_new, trained_embeddings_BERT)
#     similarity_df = pd.DataFrame(similarity_new_data)

#     index_similarity = similarity_df.idxmax(axis=1)
    
#     # i need to make sure that the indices match
#     indexes = index_similarity.tolist()
#     response = supabase.table('expenses').select("*").in_("id", indexes).execute()
#     relevant_rows = response.data
#     data_inspect = pd.DataFrame(relevant_rows)
#     # data_inspect = df_training_data.iloc[index_similarity, :].reset_index(drop=True)

#     annotation = data_inspect["category"]

#     d_output = {
#         "Date": date,
#         "Amount": amount,
#         "Narrative": desc_new_data,
#         "Categories": annotation,
#     }
#     return d_output


# def run_test_classify(test_file_name):
#     with app.test_client() as client:
#         with open(test_file_name, "rb") as file:
#             response = client.post("/api/classify", data={"file": file})
#             # print(response.get_json())


# def run_test_adminUpload(test_file_name):
#     with app.test_client() as client:
#         with open(test_file_name, "rb") as file:
#             response = client.post("/api/adminUpload", data={"file": file})
#             # print(response.get_json())


# def run_test_retrain(test_data):
#     with app.test_client() as client:
#         response = client.post("/api/retrain", json=test_data)
#         print(response.get_json())


# if __name__ == "__main__":
#     # app.run()
#     # test_file_name = "all_expenses_classified.csv"    
#     # run_test_adminUpload(test_file_name)
    
#     # test_file_name = "01_ExpenseDetails_2023 - CSVData (9).csv"    
#     # run_test_classify(test_file_name)

#     test_data = [
#         {
#             "date": "27/05/2023",
#             "debitamount": -27.88,
#             "description": "Aldi shopping",
#             "category": "Groceries",
#         },
#         {
#             "date": "26/05/2023",
#             "debitamount": -5.44,
#             "description": "Beers at the pub",
#             "category": "DinnerBars",
#         },
#         {
#             "date": "24/05/2023",
#             "debitamount": -7.58,
#             "description": "Coffee at tge beach",
#             "category": "DinnerBars",
#         },
#     ]
#     run_test_retrain(test_data)

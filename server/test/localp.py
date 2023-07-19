import requests

# HTTP unauthenticated endpoint to call
url = "http://localhost:8080"

# Set the mode to either "train" or "classify"
mode = "train"

# Read the CSV file
file_path = "/Users/fstrauf/Documents/01_code/txClassify/all_expenses_classified.csv"
csv_file = open(file_path, "rb")

files = {
    "file": csv_file
}

# Send a POST request to the endpoint
r = requests.post(url + "?mode=" + mode, files=files)

# Print the response content
print(r.content)

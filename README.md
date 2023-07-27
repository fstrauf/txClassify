

## Introduction

This is a hybrid Next.js + Python app that uses Next.js as the frontend and Flask as the API backend. One great use case of this is to write Next.js apps that use Python AI libraries on the backend.

## How It Works

```npm run dev``` spins up flask with python and nextjs.

```python3 api/index.py``` let's you test the API separately as a python scripy


```curl -m 70 -X POST "http://localhost:8080/txclassify?mode=train" \
-H "Authorization: bearer $(gcloud auth print-identity-token)" \
-F "file=@/Users/fstrauf/Documents/01_code/txClassify/all_expenses_classified.csv"


curl -m 70 -X POST "https://us-central1-txclassify.cloudfunctions.net/txclassify?mode=train" \
-H "Authorization: bearer $(gcloud auth print-identity-token)" \
-F "file=@/Users/fstrauf/Documents/01_code/txClassify/all_expenses_classified.csv"```
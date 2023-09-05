## deploy to google

```gcloud run deploy```

## run locally

Comment in port 3000 line

```
if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    app.run(debug=True, host="0.0.0.0", port=3000)
```
run flask dev server
```
export FLASK_APP=main.py
flask run --port=3001
```

run ngrok to route webhooks to your local machine

```npx ngrok http 3001```

Run this script locally
```gcloud auth application-default login
```

Get the credentials file 
```/Users/fstrauf/.config/gcloud/application_default_credentials.json```

set the environment
```export GOOGLE_APPLICATION_CREDENTIALS="/Users/fstrauf/.config/gcloud/application_default_credentials.json"
```

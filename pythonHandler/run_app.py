from app import app

if __name__ == "__main__":
    # Run the Flask app in debug mode
    app.run(host="0.0.0.0", port=5001, debug=True) 
# Transaction Categorization API Documentation

This document provides details on how to integrate with the Transaction Categorization API.

## Table of Contents

- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Base URL](#base-url)
- [Rate Limits](#rate-limits)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Training](#training)
  - [Classification](#classification)
  - [Status Check](#status-check)
  - [API Key Management](#api-key-management)
  - [User Configuration](#user-configuration)
  - [API Usage](#api-usage)

## Authentication

Authentication is handled via API keys that should be included in the header of each request.

```
X-API-Key: your_api_key
```

## Error Handling

The API returns HTTP status codes that indicate whether a request was successful or not:

- `200 OK`: Request was successful
- `400 Bad Request`: Request format is invalid
- `401 Unauthorized`: API key is missing or invalid
- `404 Not Found`: The requested resource was not found
- `500 Internal Server Error`: Server-side error

Error responses follow this format:

```json
{
  "status": "error",
  "error": "Detailed error message",
  "code": 400
}
```

## Base URL

All API requests should be made to:

```
https://api.txclassify.com
```

For testing, you can use:

```
http://localhost:3001
```

## Rate Limits

The API has rate limits of 100 requests per minute per API key. If exceeded, the API will return a 429 status code.

## Endpoints

### Health Check

Check if the API is up and running.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2023-07-14T12:34:56.789Z",
  "components": {
    "app": "healthy",
    "database": "healthy"
  }
}
```

### Training

Train the categorization model with your transaction data.

**Endpoint:** `POST /train`

**Headers:**

- `Content-Type: application/json`
- `X-API-Key: your_api_key`

**Request Body:**

```json
{
  "transactions": [
    {
      "description": "PIZZA HUT DELIVERY",
      "Category": "Food & Dining"
    },
    {
      "description": "AMAZON.COM",
      "Category": "Shopping"
    }
    // At least 10 transactions are required
  ],
  "expenseSheetId": "unique_sheet_identifier",
  "userId": "optional_user_id"
}
```

**Response:**

```json
{
  "status": "processing",
  "prediction_id": "f99essksjdrge0cnqkhvcvzrhw"
}
```

**Notes:**

- The `transactions` array must contain at least 10 items
- Each transaction must have a `description` and `Category` field
- The `expenseSheetId` is used to identify this training dataset
- Training is asynchronous. Use the `/status/{prediction_id}` endpoint to check progress

### Classification

Classify new transactions based on your trained model.

**Endpoint:** `POST /classify`

**Headers:**

- `Content-Type: application/json`
- `X-API-Key: your_api_key`

**Request Body:**

```json
{
  "transactions": ["UBER TRIP 12345", "STARBUCKS COFFEE", "AMAZON.COM PAYMENT"],
  "spreadsheetId": "unique_sheet_identifier",
  "sheetName": "new_transactions",
  "categoryColumn": "E",
  "startRow": "1"
}
```

**Response:**

```json
{
  "status": "processing",
  "prediction_id": "a88bssksjdrge0cnqkhvcvzrhs",
  "message": "Classification started. Check status endpoint for updates."
}
```

**Notes:**

- The `transactions` array contains transaction descriptions as strings
- The `spreadsheetId` must match what was used during training
- Classification is asynchronous. Use the `/status/{prediction_id}` endpoint to check progress

### Status Check

Check the status of a training or classification task.

**Endpoint:** `GET /status/{prediction_id}`

**Headers:**

- `X-API-Key: your_api_key`

**Responses:**

For in-progress tasks:

```json
{
  "status": "processing",
  "message": "Processing in progress"
}
```

For completed classification tasks:

```json
{
  "status": "completed",
  "message": "Processing completed successfully",
  "results": [
    {
      "predicted_category": "Transport",
      "similarity_score": 0.92,
      "narrative": "UBER TRIP 12345"
    },
    {
      "predicted_category": "Food & Dining",
      "similarity_score": 0.87,
      "narrative": "STARBUCKS COFFEE"
    }
  ],
  "config": {
    "categoryColumn": "E",
    "startRow": "1",
    "sheetName": "new_transactions",
    "spreadsheetId": "unique_sheet_identifier"
  }
}
```

For completed training tasks:

```json
{
  "status": "completed",
  "message": "Training completed successfully!"
}
```

For failed tasks:

```json
{
  "status": "error",
  "error": "Detailed error message",
  "code": 500
}
```

### API Key Management

Create or retrieve an API key.

**GET Endpoint:** `GET /api-key?userId=user_identifier`

**Headers:**

- `X-API-Key: your_api_key` (optional - for validation)

**Response:**

```json
{
  "status": "success",
  "user_id": "user_identifier",
  "api_key": "api_key_value"
}
```

**POST Endpoint:** `POST /api-key`

**Headers:**

- `Content-Type: application/json`

**Request Body:**

```json
{
  "userId": "user_identifier"
}
```

**Response:**

```json
{
  "status": "success",
  "user_id": "user_identifier",
  "api_key": "newly_generated_api_key"
}
```

### User Configuration

Retrieve user configuration details.

**Endpoint:** `GET /user-config?userId=user_identifier`

**Headers:**

- `X-API-Key: your_api_key` (optional)

**Response:**

```json
{
  "userId": "user_identifier",
  "categorisationRange": "A:Z",
  "categorisationTab": "SheetName",
  "columnOrderCategorisation": {
    "categoryColumn": "E",
    "descriptionColumn": "C"
  },
  "api_key": "api_key_value"
}
```

### API Usage

Get API usage statistics for your account.

**Endpoint:** `GET /api-usage`

**Headers:**

- `X-API-Key: your_api_key`

**Response:**

```json
{
  "user_id": "user_identifier",
  "usage": {
    "total_requests": 156,
    "total_classifications": 89,
    "total_training": 5,
    "current_month_requests": 42
  },
  "limits": {
    "monthly_limit": 1000,
    "rate_limit": "100 per minute"
  }
}
```

## Object Definitions

### Transaction Object

Used in training requests:

```json
{
  "description": "AMAZON.COM PAYMENT",
  "Category": "Shopping"
}
```

Required fields:

- `description`: String containing the transaction description
- `Category`: String containing the category label

### Classification Result Object

Returned in completed classification responses:

```json
{
  "predicted_category": "Transport",
  "similarity_score": 0.92,
  "narrative": "UBER TRIP 12345"
}
```

Fields:

- `predicted_category`: String containing the predicted category
- `similarity_score`: Number between 0 and 1 indicating confidence
- `narrative`: Original transaction description

## Example Workflows

### Complete Training and Classification Flow

1. Train the model with labeled data:

```bash
curl -X POST https://api.txclassify.com/train \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "transactions": [
      {"description": "UBER EATS", "Category": "Food & Dining"},
      // ... at least 10 transactions
    ],
    "expenseSheetId": "sheet123"
  }'
```

2. Check training status:

```bash
curl https://api.txclassify.com/status/prediction_id_from_step_1 \
  -H "X-API-Key: your_api_key"
```

3. Once training is complete, classify new transactions:

```bash
curl -X POST https://api.txclassify.com/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "transactions": [
      "STARBUCKS COFFEE",
      "NETFLIX SUBSCRIPTION"
    ],
    "spreadsheetId": "sheet123"
  }'
```

4. Check classification results:

```bash
curl https://api.txclassify.com/status/prediction_id_from_step_3 \
  -H "X-API-Key: your_api_key"
```

# T5 Sentiment Model Deployment on AWS

This repository contains the complete code to deploy a custom T5 Sentiment-Gate model to AWS using SageMaker Serverless Inference, Lambda, and API Gateway.

## Architecture

1.  **Model Packaging**: The custom T5 model (`t5_sentiment_gate.py`, `pytorch_model.bin`, etc.) is packaged into a `model.tar.gz` with a custom inference script.
2.  **SageMaker Serverless Inference**: The model is deployed to a serverless endpoint (pay-per-request).
3.  **AWS Lambda**: A Lambda function acts as a proxy, receiving requests, invoking the SageMaker endpoint, and returning the result.
4.  **API Gateway**: An HTTP API exposes a public `POST /predict` route that triggers the Lambda function.

## Prerequisites

*   AWS CLI installed and configured (`aws configure`).
*   Python 3.8+ installed.
*   `boto3`, `sagemaker`, `requests` libraries installed.
*   **Model Files**: Ensure your model files (`pytorch_model.bin`, `config.json`, `t5_sentiment_gate.py`, etc.) are in the parent directory or update `package_model.py` to point to them.

## Setup & Deployment

### 1. Install Dependencies

```bash
pip install boto3 sagemaker requests
```

### 2. Package the Model

Run the packaging script to create `model.tar.gz`.

```bash
python package_model.py
```

This will look for model files in the parent directory (or configured path) and create `model.tar.gz` in the current directory.

### 3. Deploy to SageMaker

Deploy the model to a Serverless Endpoint.

```bash
python deploy_sagemaker.py
```

*   **Note**: This script attempts to use your default SageMaker execution role. If it fails, you may need to specify a valid `ROLE_ARN` in the script.
*   This step takes a few minutes.

### 4. Create API Gateway & Lambda

Create the Lambda function and API Gateway route.

```bash
python create_api_gateway.py
```

*   This script creates an IAM role for Lambda, zips the `lambda_inference.py` code, creates the function, and sets up the API Gateway.
*   It will output the **API URL** at the end.

### 5. Test the API

Use the generated API URL to test the deployment.

```bash
python test_api.py <YOUR_API_URL>
```

Example:
```bash
python test_api.py https://xyz.execute-api.us-east-1.amazonaws.com/predict
```

## Folder Structure

*   `code/`: Contains `inference.py` (SageMaker entry point) and `requirements.txt`.
*   `package_model.py`: Script to bundle model artifacts.
*   `deploy_sagemaker.py`: Script to deploy SageMaker Serverless Endpoint.
*   `lambda_inference.py`: Code for the Lambda function.
*   `create_api_gateway.py`: Script to setup Lambda and API Gateway.
*   `test_api.py`: Test script.

## Customization

*   **Model Class**: The inference script imports `T5ForSentimentClassification` from `t5_sentiment_gate.py`. Ensure this file is included in the package.
*   **Serverless Config**: Memory and Concurrency are set in `deploy_sagemaker.py`.

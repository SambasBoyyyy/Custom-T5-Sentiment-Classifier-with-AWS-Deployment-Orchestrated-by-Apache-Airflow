import json
import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variable for the SageMaker endpoint name
ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 't5-sentiment-serverless-endpoint')
# AWS_DEFAULT_REGION is automatically set by Lambda runtime
REGION = os.environ.get('AWS_DEFAULT_REGION', os.environ.get('AWS_REGION', 'us-east-1'))

runtime = boto3.client('runtime.sagemaker', region_name=REGION)

def lambda_handler(event, context):
    """
    Lambda handler to invoke SageMaker endpoint
    """
    logger.info("Received event: %s", json.dumps(event))
    
    try:
        # Parse input
        # API Gateway HTTP API (v2) or REST API (v1) structure might differ
        # We assume a body containing JSON
        body_str = event.get('body', '{}')
        if not body_str:
            body_str = '{}'
            
        # Handle base64 encoding if necessary (API Gateway sometimes encodes body)
        if event.get('isBase64Encoded', False):
            import base64
            body_str = base64.b64decode(body_str).decode('utf-8')
            
        body = json.loads(body_str)
        
        # Extract text
        text = body.get('text', '')
        if not text:
            # Fallback for testing or different input structure
            text = body.get('inputs', '')
            
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing "text" field in request body'})
            }

        logger.info(f"Invoking endpoint {ENDPOINT_NAME} with text: {text}")
        
        # Invoke SageMaker Endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps({"text": text})
        )
        
        # Parse response
        response_body = response['Body'].read().decode('utf-8')
        result = json.loads(response_body)
        
        logger.info(f"Prediction result: {result}")
        
        # Return response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({'error': str(e)})
        }

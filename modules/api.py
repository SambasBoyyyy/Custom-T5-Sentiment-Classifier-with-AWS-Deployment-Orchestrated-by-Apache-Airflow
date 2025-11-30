"""
API Setup

Creates Lambda function and API Gateway for public access.
"""

import zipfile
import tempfile
from pathlib import Path
import boto3
from .config import config
from .utils import print_step


def create_lambda(role_arn: str, endpoint_name: str) -> str:
    """
    Create Lambda function that calls SageMaker endpoint.
    
    Args:
        role_arn: IAM role ARN
        endpoint_name: SageMaker endpoint name
        
    Returns:
        Lambda function ARN
    """
    print_step(4, "Creating Lambda Function")
    
    lambda_client = boto3.client('lambda', region_name=config.region)
    
    # Lambda code - simple handler that calls SageMaker
    lambda_code = f'''import json
import boto3

sagemaker = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    """Handle API requests and call SageMaker"""
    try:
        # Parse request
        body = json.loads(event.get('body', '{{}}'))
        text = body.get('text', '')
        
        if not text:
            return {{
                'statusCode': 400,
                'body': json.dumps({{'error': 'Missing text field'}})
            }}
        
        # Call SageMaker endpoint
        response = sagemaker.invoke_endpoint(
            EndpointName='{endpoint_name}',
            ContentType='application/json',
            Body=json.dumps({{'text': text}})
        )
        
        # Return result
        result = json.loads(response['Body'].read())
        
        return {{
            'statusCode': 200,
            'headers': {{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }},
            'body': json.dumps(result)
        }}
        
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}})
        }}
'''
    
    # Create deployment package
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "function.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr('lambda_function.py', lambda_code)
        
        with open(zip_path, 'rb') as f:
            zip_content = f.read()
        
        # Create or update function
        try:
            response = lambda_client.create_function(
                FunctionName=config.lambda_name,
                Runtime='python3.10',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_content},
                Timeout=120,  # 2 minutes for GPU inference
                MemorySize=512,
                Description='T5 Sentiment Classification API handler'
            )
            print(f"✓ Created Lambda function: {config.lambda_name}")
        except lambda_client.exceptions.ResourceConflictException:
            lambda_client.update_function_code(
                FunctionName=config.lambda_name,
                ZipFile=zip_content
            )
            # Wait a moment for update
            import time
            time.sleep(2)
            response = lambda_client.get_function(FunctionName=config.lambda_name)
            print(f"✓ Updated Lambda function: {config.lambda_name}")
        
        # Add permission for API Gateway
        try:
            lambda_client.add_permission(
                FunctionName=config.lambda_name,
                StatementId='api-gateway-invoke',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com'
            )
        except:
            pass  # Permission already exists
        
        return response['FunctionArn'] if 'FunctionArn' in response else response['Configuration']['FunctionArn']


def create_api_gateway(lambda_arn: str) -> str:
    """
    Create API Gateway with /classify endpoint.
    
    Args:
        lambda_arn: Lambda function ARN
        
    Returns:
        API URL
    """
    print_step(5, "Creating API Gateway")
    
    apigateway = boto3.client('apigateway', region_name=config.region)
    
    # Delete old API if exists
    apis = apigateway.get_rest_apis()
    for api in apis['items']:
        if api['name'] == config.api_name:
            apigateway.delete_rest_api(restApiId=api['id'])
            print(f"  Deleted old API")
    
    # Create new API
    api = apigateway.create_rest_api(
        name=config.api_name,
        description='T5 Sentiment Classification API',
        endpointConfiguration={'types': ['REGIONAL']}
    )
    api_id = api['id']
    print(f"✓ Created API: {config.api_name}")
    
    # Get root resource
    resources = apigateway.get_resources(restApiId=api_id)
    root_id = resources['items'][0]['id']
    
    # Create /classify resource
    resource = apigateway.create_resource(
        restApiId=api_id,
        parentId=root_id,
        pathPart='classify'
    )
    resource_id = resource['id']
    
    # Create POST method
    apigateway.put_method(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='POST',
        authorizationType='NONE'
    )
    
    # Integrate with Lambda
    lambda_uri = f"arn:aws:apigateway:{config.region}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"
    
    apigateway.put_integration(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod='POST',
        type='AWS_PROXY',
        integrationHttpMethod='POST',
        uri=lambda_uri
    )
    
    # Deploy API
    apigateway.create_deployment(
        restApiId=api_id,
        stageName='prod'
    )
    
    # Build URL
    api_url = f"https://{api_id}.execute-api.{config.region}.amazonaws.com/prod/classify"
    print(f"✓ API deployed at: {api_url}")
    
    return api_url


def create_api(endpoint_name: str, role_arn: str) -> str:
    """
    Create public API for the endpoint.
    Wrapper function called by deploy.py.
    """
    lambda_arn = create_lambda(role_arn, endpoint_name)
    return create_api_gateway(lambda_arn)

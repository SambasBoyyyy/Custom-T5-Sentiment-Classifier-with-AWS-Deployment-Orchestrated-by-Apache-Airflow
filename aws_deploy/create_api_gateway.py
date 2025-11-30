import boto3
import json
import time
import zipfile
import os

# --- CONFIGURATION ---
REGION = boto3.Session().region_name or "us-east-1"
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

LAMBDA_FUNCTION_NAME = "t5-sentiment-lambda"
API_NAME = "t5-sentiment-api"
ENDPOINT_NAME = "t5-sentiment-serverless-endpoint" # Must match deploy_sagemaker.py

# IAM Role for Lambda
ROLE_NAME = "T5SentimentLambdaRole"
POLICY_ARN_SAGEMAKER = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess" # Or specific invoke permission
POLICY_ARN_BASIC = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

def create_lambda_role():
    iam = boto3.client("iam", region_name=REGION)
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        print(f"Creating IAM Role {ROLE_NAME}...")
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy)
        )
        role_arn = role['Role']['Arn']
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Role {ROLE_NAME} already exists.")
        role = iam.get_role(RoleName=ROLE_NAME)
        role_arn = role['Role']['Arn']

    print("Attaching policies...")
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=POLICY_ARN_SAGEMAKER)
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=POLICY_ARN_BASIC)
    
    print("Waiting for role propagation...")
    time.sleep(10) # Wait for role to be ready
    return role_arn

def create_lambda_function(role_arn):
    lambda_client = boto3.client("lambda", region_name=REGION)
    
    # Zip the code
    zip_filename = "lambda_function.zip"
    print(f"Zipping lambda code to {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w') as z:
        z.write(os.path.join(os.path.dirname(__file__), "lambda_inference.py"), arcname="lambda_function.py")
        
    with open(zip_filename, 'rb') as f:
        zipped_code = f.read()
        
    # Retry loop for handling "ResourceConflictException" (update in progress)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Creating Lambda function {LAMBDA_FUNCTION_NAME} (Attempt {attempt+1}/{max_retries})...")
            lambda_client.create_function(
                FunctionName=LAMBDA_FUNCTION_NAME,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zipped_code},
                Environment={
                    'Variables': {
                        'ENDPOINT_NAME': ENDPOINT_NAME
                    }
                },
                Timeout=30,
                MemorySize=128
            )
            break # Success
        except lambda_client.exceptions.ResourceConflictException as e:
            if "already exists" in str(e):
                print(f"Function {LAMBDA_FUNCTION_NAME} already exists. Updating code...")
                try:
                    lambda_client.update_function_code(
                        FunctionName=LAMBDA_FUNCTION_NAME,
                        ZipFile=zipped_code
                    )
                    # Wait for update to propagate before updating config
                    time.sleep(5)
                    lambda_client.update_function_configuration(
                        FunctionName=LAMBDA_FUNCTION_NAME,
                        Environment={
                            'Variables': {
                                'ENDPOINT_NAME': ENDPOINT_NAME
                            }
                        }
                    )
                    break # Success
                except lambda_client.exceptions.ResourceConflictException as e2:
                    if "update is in progress" in str(e2):
                        print("Update in progress. Waiting 10 seconds...")
                        time.sleep(10)
                        continue
                    else:
                        raise e2
            else:
                raise e
        except Exception as e:
            print(f"Error creating/updating lambda: {e}")
            raise e
    
    # Clean up zip
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
    
    # Get ARN
    func = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
    return func['Configuration']['FunctionArn']

def create_api_gateway(lambda_arn):
    apigateway = boto3.client("apigatewayv2", region_name=REGION)
    
    print(f"Creating API Gateway {API_NAME}...")
    # Check if exists (simple check by name, not robust but okay for script)
    apis = apigateway.get_apis()
    api_id = None
    for item in apis['Items']:
        if item['Name'] == API_NAME:
            api_id = item['ApiId']
            print(f"API {API_NAME} already exists with ID {api_id}")
            break
            
    if not api_id:
        api = apigateway.create_api(
            Name=API_NAME,
            ProtocolType='HTTP',
            CorsConfiguration={
                'AllowOrigins': ['*'],
                'AllowMethods': ['POST', 'OPTIONS'],
                'AllowHeaders': ['Content-Type']
            }
        )
        api_id = api['ApiId']
        print(f"Created API with ID {api_id}")

    # Create Integration
    print("Creating Integration...")
    integration = apigateway.create_integration(
        ApiId=api_id,
        IntegrationType='AWS_PROXY',
        IntegrationUri=lambda_arn,
        PayloadFormatVersion='2.0'
    )
    integration_id = integration['IntegrationId']

    # Create Route
    print("Creating Route POST /predict...")
    try:
        apigateway.create_route(
            ApiId=api_id,
            RouteKey='POST /predict',
            Target=f'integrations/{integration_id}'
        )
    except apigateway.exceptions.ConflictException:
        print("Route POST /predict already exists.")

    # Create Stage (default is $default usually auto-created, but let's ensure)
    # HTTP APIs usually have a $default stage that auto-deploys.
    
    # Add permission for API Gateway to invoke Lambda
    lambda_client = boto3.client("lambda", region_name=REGION)
    try:
        lambda_client.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId=f"apigateway-invoke-{int(time.time())}",
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{REGION}:{ACCOUNT_ID}:{api_id}/*/*/predict"
        )
        print("Added Lambda permission for API Gateway.")
    except lambda_client.exceptions.ResourceConflictException:
        print("Lambda permission already exists.")

    endpoint_url = f"https://{api_id}.execute-api.{REGION}.amazonaws.com/predict" # Note: Default stage usually doesn't need stage name in path if configured right, but often it's https://id.execute-api.region.amazonaws.com/default/predict or similar.
    # Let's get the endpoint
    api_info = apigateway.get_api(ApiId=api_id)
    endpoint = api_info.get('ApiEndpoint', f"https://{api_id}.execute-api.{REGION}.amazonaws.com")
    
    print("\n" + "="*50)
    print("DEPLOYMENT COMPLETE")
    print("="*50)
    print(f"API URL: {endpoint}/predict")
    print("Test with:")
    print(f'curl -X POST {endpoint}/predict -H "Content-Type: application/json" -d \'{{"text": "I love this product!"}}\'')
    
    return f"{endpoint}/predict"

if __name__ == "__main__":
    role_arn = create_lambda_role()
    lambda_arn = create_lambda_function(role_arn)
    api_url = create_api_gateway(lambda_arn)

import boto3
import sagemaker
import os
import time

# --- CONFIGURATION ---
REGION = boto3.Session().region_name or "us-east-1"
# Get Account ID
sts = boto3.client("sts")
ACCOUNT_ID = sts.get_caller_identity()["Account"]

# Bucket for model artifacts
BUCKET_NAME = f"t5-sentiment-model-{ACCOUNT_ID}-{REGION}"
PREFIX = "t5-sentiment-serverless"

# Role ARN - Using the role created by setup_iam_role.py
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/AmazonSageMaker-ExecutionRole"

# Model and Endpoint Names
MODEL_NAME = "t5-sentiment-gate-model"
ENDPOINT_CONFIG_NAME = "t5-sentiment-serverless-config"
ENDPOINT_NAME = "t5-sentiment-serverless-endpoint"

# Container Image - Use CPU for serverless (GPU not supported in serverless)
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04"

# Deployment mode: 'serverless' or 'realtime'
# Serverless is pay-per-request, realtime is always-on
DEPLOYMENT_MODE = os.environ.get('DEPLOYMENT_MODE', 'serverless')
INSTANCE_TYPE = os.environ.get('INSTANCE_TYPE', 'ml.g4dn.xlarge')  # Only used for realtime mode

def deploy():
    print(f"Deploying to Region: {REGION}")
    print(f"Account ID: {ACCOUNT_ID}")
    print(f"Deployment Mode: {DEPLOYMENT_MODE}")
    
    # Initialize SageMaker Session
    sess = sagemaker.Session()
    
    # 1. Upload model.tar.gz to S3
    model_tar_path = os.path.join(os.path.dirname(__file__), "model.tar.gz")
    if not os.path.exists(model_tar_path):
        print(f"ERROR: {model_tar_path} not found. Run package_model.py first.")
        return

    print(f"Uploading {model_tar_path} to s3://{BUCKET_NAME}/{PREFIX}/model.tar.gz ...")
    # Create bucket if not exists
    s3 = boto3.client('s3', region_name=REGION)
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(Bucket=BUCKET_NAME, CreateBucketConfiguration={'LocationConstraint': REGION})
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    except Exception as e:
        print(f"Error creating bucket: {e}")

    model_data = sess.upload_data(
        path=model_tar_path,
        bucket=BUCKET_NAME,
        key_prefix=PREFIX
    )
    print(f"Model uploaded to: {model_data}")

    # 2. Create SageMaker Model
    print("Creating SageMaker Model object...")
    
    sm_client = boto3.client("sagemaker", region_name=REGION)
    
    # Check if model exists and delete if so
    try:
        sm_client.delete_model(ModelName=MODEL_NAME)
        print(f"Deleted existing model: {MODEL_NAME}")
        time.sleep(2)
    except:
        pass

    primary_container = {
        "Image": IMAGE_URI,
        "ModelDataUrl": model_data,
        "Environment": {
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": model_data
        }
    }

    print(f"Creating model {MODEL_NAME} with role {ROLE_ARN}...")
    try:
        sm_client.create_model(
            ModelName=MODEL_NAME,
            ExecutionRoleArn=ROLE_ARN,
            PrimaryContainer=primary_container
        )
        print(f"Model {MODEL_NAME} created successfully!")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Ensure you have a valid SageMaker execution role and update ROLE_ARN in the script if needed.")
        return

    # 3. Create Endpoint Config (Serverless or Real-time)
    print(f"Creating {DEPLOYMENT_MODE.capitalize()} Endpoint Config...")
    
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
        print(f"Deleted existing endpoint config: {ENDPOINT_CONFIG_NAME}")
        time.sleep(2)
    except:
        pass

    try:
        if DEPLOYMENT_MODE == 'serverless':
            # Serverless configuration - reduced to 3072 MB to fit quota
            sm_client.create_endpoint_config(
                EndpointConfigName=ENDPOINT_CONFIG_NAME,
                ProductionVariants=[
                    {
                        "VariantName": "AllTraffic",
                        "ModelName": MODEL_NAME,
                        "ServerlessConfig": {
                            "MemorySizeInMB": 3072,  # Max allowed by account quota
                            "MaxConcurrency": 10
                        }
                    }
                ]
            )
        else:
            # Real-time configuration with GPU instance
            sm_client.create_endpoint_config(
                EndpointConfigName=ENDPOINT_CONFIG_NAME,
                ProductionVariants=[
                    {
                        "VariantName": "AllTraffic",
                        "ModelName": MODEL_NAME,
                        "InstanceType": INSTANCE_TYPE,
                        "InitialInstanceCount": 1
                    }
                ]
            )
        print(f"Endpoint config {ENDPOINT_CONFIG_NAME} created successfully!")
    except Exception as e:
        print(f"Error creating endpoint config: {e}")
        if "serverless" in str(e).lower() or "quota" in str(e).lower():
            print("\n" + "="*60)
            print("SERVERLESS QUOTA ISSUE DETECTED")
            print("="*60)
            print("Your AWS account may not have access to Serverless Inference.")
            print("This is common for new accounts or certain regions.")
            print("\nTo deploy using a real-time endpoint instead, run:")
            print("  set DEPLOYMENT_MODE=realtime")
            print("  python aws_deploy/deploy_sagemaker.py")
            print("\nOr on Linux/Mac:")
            print("  DEPLOYMENT_MODE=realtime python aws_deploy/deploy_sagemaker.py")
        return

    # 4. Create Endpoint
    print(f"Creating Endpoint {ENDPOINT_NAME}...")
    try:
        sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Deleted existing endpoint: {ENDPOINT_NAME}")
        # Wait for deletion
        time.sleep(10)
    except:
        pass

    try:
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME
        )
        print(f"Endpoint {ENDPOINT_NAME} creation started!")
    except Exception as e:
        print(f"Error creating endpoint: {e}")
        return

    print("Waiting for endpoint deployment (this takes a few minutes)...")
    
    try:
        waiter = sm_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=ENDPOINT_NAME, WaiterConfig={'Delay': 30, 'MaxAttempts': 60})
        
        print("\n" + "="*60)
        print("DEPLOYMENT SUCCESSFUL!")
        print("="*60)
        print(f"Endpoint Name: {ENDPOINT_NAME}")
        print(f"Endpoint ARN: {sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)['EndpointArn']}")
        print(f"Deployment Mode: {DEPLOYMENT_MODE}")
        print("\nNext step: Run the Lambda and API Gateway setup:")
        print("  python aws_deploy/create_api_gateway.py")
    except Exception as e:
        print(f"Error waiting for endpoint: {e}")
        print("Check the AWS Console for more details.")

if __name__ == "__main__":
    deploy()

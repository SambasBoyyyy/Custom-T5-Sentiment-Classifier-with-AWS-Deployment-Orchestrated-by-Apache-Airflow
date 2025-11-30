"""
AWS Infrastructure Setup

Creates the necessary AWS resources:
- IAM role with permissions
- S3 bucket for model storage
"""

import json
import time
import boto3
from botocore.exceptions import ClientError
from .config import config
from .utils import print_step


def create_iam_role() -> str:
    """
    Create IAM role with permissions for SageMaker and Lambda.
    
    Returns:
        Role ARN
    """
    print_step(1, "Creating IAM Role")
    
    iam = boto3.client('iam', region_name=config.region)
    
    # Trust policy - allows SageMaker and Lambda to assume this role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": [
                        "sagemaker.amazonaws.com",
                        "lambda.amazonaws.com"
                    ]
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # Create role
    try:
        iam.create_role(
            RoleName=config.role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for T5 deployment (SageMaker + Lambda)"
        )
        print(f"✓ Created role: {config.role_name}")
    except ClientError as e:
        if 'EntityAlreadyExists' in str(e):
            print(f"✓ Role already exists: {config.role_name}")
        else:
            raise
    
    # Attach managed policies
    # These give the role permissions to use AWS services
    policies = [
        'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',  # SageMaker
        'arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole',  # Lambda logs
        'arn:aws:iam::aws:policy/AmazonS3FullAccess'  # S3 storage
    ]
    
    for policy_arn in policies:
        try:
            iam.attach_role_policy(
                RoleName=config.role_name,
                PolicyArn=policy_arn
            )
        except ClientError:
            pass  # Already attached
    
    print(f"✓ Attached policies to role")
    
    # Wait for role to propagate (AWS eventual consistency)
    print("  Waiting for role to be ready...")
    time.sleep(10)
    
    return config.role_arn


def create_s3_bucket() -> str:
    """
    Create S3 bucket for storing the model.
    
    Returns:
        Bucket name
    """
    print_step(2, "Creating S3 Bucket")
    
    s3 = boto3.client('s3', region_name=config.region)
    
    try:
        # Create bucket
        if config.region == 'us-east-1':
            # us-east-1 doesn't need LocationConstraint
            s3.create_bucket(Bucket=config.bucket_name)
        else:
            s3.create_bucket(
                Bucket=config.bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': config.region
                }
            )
        print(f"✓ Created bucket: {config.bucket_name}")
        
    except ClientError as e:
        if 'BucketAlreadyOwnedByYou' in str(e):
            print(f"✓ Bucket already exists: {config.bucket_name}")
        else:
            raise
    
    return config.bucket_name


def setup_aws():
    """
    Setup all AWS resources.
    
    Returns:
        tuple: (bucket_name, role_arn)
    """
    role_arn = create_iam_role()
    bucket_name = create_s3_bucket()
    return bucket_name, role_arn


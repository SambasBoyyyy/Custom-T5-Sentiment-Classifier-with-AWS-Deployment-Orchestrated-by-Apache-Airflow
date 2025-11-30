import boto3
import json
import time

# --- CONFIGURATION ---
REGION = boto3.Session().region_name or "us-east-1"
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET_NAME = f"t5-sentiment-model-{ACCOUNT_ID}-{REGION}"

ROLE_NAME = "AmazonSageMaker-ExecutionRole"
POLICY_NAME = "SageMakerS3AccessPolicy"

def create_sagemaker_role():
    """Create or update SageMaker execution role with proper permissions"""
    iam = boto3.client("iam", region_name=REGION)
    
    # Trust relationship policy for SageMaker
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # S3 access policy for the specific bucket
    s3_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    f"arn:aws:s3:::{BUCKET_NAME}",
                    f"arn:aws:s3:::{BUCKET_NAME}/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket",
                    "s3:GetBucketLocation"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage"
                ],
                "Resource": "*"
            }
        ]
    }
    
    # Try to create the role
    try:
        print(f"Creating IAM role: {ROLE_NAME}...")
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SageMaker execution role for T5 sentiment model"
        )
        role_arn = role['Role']['Arn']
        print(f"Created role: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Role {ROLE_NAME} already exists.")
        role = iam.get_role(RoleName=ROLE_NAME)
        role_arn = role['Role']['Arn']
        
        # Update trust policy
        print("Updating trust relationship policy...")
        iam.update_assume_role_policy(
            RoleName=ROLE_NAME,
            PolicyDocument=json.dumps(trust_policy)
        )
    
    # Create or update inline policy
    print(f"Attaching S3 access policy...")
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName=POLICY_NAME,
        PolicyDocument=json.dumps(s3_policy)
    )
    
    # Attach AWS managed policies
    managed_policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    ]
    
    for policy_arn in managed_policies:
        try:
            print(f"Attaching managed policy: {policy_arn}")
            iam.attach_role_policy(
                RoleName=ROLE_NAME,
                PolicyArn=policy_arn
            )
        except iam.exceptions.LimitExceededException:
            print(f"Policy {policy_arn} already attached.")
        except Exception as e:
            print(f"Note: {e}")
    
    print("\nWaiting for IAM role to propagate (10 seconds)...")
    time.sleep(10)
    
    print("\n" + "="*60)
    print("IAM ROLE SETUP COMPLETE")
    print("="*60)
    print(f"Role ARN: {role_arn}")
    print(f"Role Name: {ROLE_NAME}")
    print(f"Bucket: {BUCKET_NAME}")
    print("\nYou can now run: python aws_deploy/deploy_sagemaker.py")
    
    return role_arn

if __name__ == "__main__":
    create_sagemaker_role()

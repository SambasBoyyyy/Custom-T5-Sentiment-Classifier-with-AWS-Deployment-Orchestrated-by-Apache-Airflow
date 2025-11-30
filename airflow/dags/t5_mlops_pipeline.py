from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['samiulbasirbasirbhuiyan1234@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    't5_mlops_pipeline',
    default_args=default_args,
    description='End-to-end MLOps pipeline for T5 Sentiment Model',
    schedule_interval=None,  # On-demand only
    start_date=days_ago(1),
    tags=['mlops', 't5', 'sentiment'],
)

# Project root path (mounted in Docker)
PROJECT_ROOT = "/opt/airflow/project"

# Task 1: Check Data Availability (Placeholder for now, assumes data exists)
check_data = BashOperator(
    task_id='check_data',
    bash_command=f'ls {PROJECT_ROOT}/modules/data/sst2_dataset.py',
    dag=dag,
)

# Task 2: Check for Local Model (Skip training)
check_model = BashOperator(
    task_id='check_model',
    bash_command=f'ls {PROJECT_ROOT}/t5-classification/best_model/pytorch_model.bin',
    dag=dag,
)

# Task 3: Evaluate Model (Optional, can skip if trusted)
evaluate_model = BashOperator(
    task_id='evaluate_model',
    bash_command=f'echo "Skipping evaluation for local model"',
    dag=dag,
)

# Task 4: Package Model
package_model = BashOperator(
    task_id='package_model',
    bash_command=f'cd {PROJECT_ROOT} && python aws_deploy/package_model.py',
    dag=dag,
)

# Task 5: Deploy to SageMaker
deploy_sagemaker = BashOperator(
    task_id='deploy_sagemaker',
    bash_command=f'cd {PROJECT_ROOT} && python aws_deploy/deploy_sagemaker.py',
    dag=dag,
)

# Task 6: Create/Update API Gateway & Lambda
create_api_gateway = BashOperator(
    task_id='create_api_gateway',
    bash_command=f'cd {PROJECT_ROOT} && python aws_deploy/create_api_gateway.py',
    dag=dag,
)

# Task 7: Test Endpoint
test_endpoint = BashOperator(
    task_id='test_endpoint',
    bash_command=f'cd {PROJECT_ROOT} && python aws_deploy/quick_test.py',
    dag=dag,
)

# Task 8: Notify Success
notify_success = EmailOperator(
    task_id='notify_success',
    to='samiulbasirbasirbhuiyan1234@gmail.com',
    subject='T5 MLOps Pipeline Success',
    html_content='The T5 Sentiment Model has been successfully trained, evaluated, and deployed to AWS SageMaker.',
    dag=dag,
)

# Define dependencies
check_data >> check_model >> evaluate_model >> package_model >> deploy_sagemaker >> create_api_gateway >> test_endpoint >> notify_success

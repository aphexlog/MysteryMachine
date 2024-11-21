import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
import logging
from mypy_boto3_s3 import S3Client
import boto3
import csv

logger = logging.getLogger()

# Initialize an S3 client
s3: S3Client = boto3.client("s3")  # type: ignore

# Initialize a SageMaker session
session = sagemaker.Session()


def create_bucket(bucket: str) -> None:
    try:
        s3.create_bucket(Bucket=bucket)
    except Exception as e:
        logger.error(e)


def upload_data(bucket: str, path: str, data: str) -> None:
    try:
        s3.put_object(Bucket=bucket, Key=path, Body=data)
    except Exception as e:
        logger.error(e)


# Create the IP Insights estimator
def create_training_job():
    # Get the IP Insights container image
    container = get_image_uri(session.boto_region_name, "ipinsights")

    # Get the execution role - this is used to give SageMaker access to your AWS resources
    role = get_execution_role()

    estimator = sagemaker.estimator.Estimator(
        container,
        role,
        instance_count=1,  # Number of instances to use for training
        instance_type="ml.c5.xlarge",  # Instance type for training
        output_path="s3://your-bucket/output",  # Output path for model artifacts
        sagemaker_session=session,
        # Hyperparameters for IP Insights
        hyperparameters={
            "num_entity_vectors": "10000",  # Number of entity embeddings
            "vector_dim": "128",  # Size of embeddings
            "batch_size": "1000",  # Training batch size
            "epochs": "10",  # Number of training epochs
            "learning_rate": "0.001",  # Learning rate
            "num_ip_addresses": "10000",  # Maximum number of IP addresses to track
        },
    )

    return estimator


# Upload training and validation data to S3
def main():
    bucket_name = "ipinsights-86589a88-8765-41e7-9019-86560161e6e2"
    create_bucket(bucket_name)

    training_path = "trianing"
    with open("example_training_1.csv", "rb") as f:
        data = f.read().decode("utf-8")
    upload_data(bucket_name, training_path, data)

    with open("example_validation_1.csv", "rb") as f:
        data = f.read().decode("utf-8")
    validation_path = "validation"
    upload_data(bucket_name, validation_path, data)

    job = create_training_job()
    # Specify the data channels for training
    job.fit(
        {
            "train": f"s3://{bucket_name}/{training_path}",
            "validation": f"s3://{bucket_name}/{validation_path}",
        }
    )


if __name__ == "__main__":
    main()

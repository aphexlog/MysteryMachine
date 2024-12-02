import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
import logging
from mypy_boto3_s3 import S3Client
import boto3
import json

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


def create_role(role_name: str) -> str:
    iam = boto3.client("iam")
    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        # Attempt to create the role
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        )
    except iam.exceptions.EntityAlreadyExistsException:
        logger.info(f"Role {role_name} already exists")
        try:
            # Attempt to retrieve the existing role
            role = iam.get_role(RoleName=role_name)
        except Exception as e:
            logger.error(f"Failed to retrieve existing role: {e}")
            raise RuntimeError("Role exists but could not be retrieved.")
    except Exception as e:
        logger.error(f"Failed to create role: {e}")
        raise RuntimeError("Role creation failed.")

    # Add policies to the role
    try:
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        )
    except Exception as e:
        logger.error(f"Failed to attach SageMaker policy: {e}")
        raise RuntimeError("Failed to attach SageMaker policy to role.")

    try:
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
        )
    except Exception as e:
        logger.error(f"Failed to attach S3 policy: {e}")
        raise RuntimeError("Failed to attach S3 policy to role.")

    # Ensure `role` has the expected structure
    if not role or "Role" not in role or "Arn" not in role["Role"]:
        logger.error(f"Role creation or retrieval returned unexpected result: {role}")
        raise RuntimeError("Role creation or retrieval failed with unexpected result.")

    return role["Role"]["Arn"]


# Create the IP Insights estimator
def create_training_artifact(output_path: str) -> sagemaker.estimator.Estimator:
    # Get the IP Insights container image
    container = get_image_uri(session.boto_region_name, "ipinsights")

    # Get the execution role - this is used to give SageMaker access to your AWS resources
    # role = get_execution_role()
    role = create_role("ipinsights-role")
    print(f"role: {role}")

    estimator = sagemaker.estimator.Estimator(
        container,
        role,
        instance_count=1,  # Number of instances to use for training
        instance_type="ml.c5.xlarge",  # Instance type for training
        output_path=output_path,  # S3 path for saving model artifacts
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

    training_path = "trianing.csv"
    with open("processed_training.csv", "rb") as f:
        data = f.read().decode("utf-8")
    upload_data(bucket_name, training_path, data)

    with open("processed_validation.csv", "rb") as f:
        data = f.read().decode("utf-8")
    validation_path = "validation.csv"
    upload_data(bucket_name, validation_path, data)

    job = create_training_artifact(f"s3://{bucket_name}/output")
    # Specify the data channels for training
    print(f"s3://{bucket_name}/{training_path}")
    print(f"s3://{bucket_name}/{validation_path}")
    job.fit(
        {
            "train": f"s3://{bucket_name}/{training_path}",
            "validation": f"s3://{bucket_name}/{validation_path}",
        }
    )


if __name__ == "__main__":
    main()

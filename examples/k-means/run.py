from sagemaker.inputs import TrainingInput
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


def create_training_artifact(output_path: str) -> sagemaker.estimator.Estimator:
    container = get_image_uri(session.boto_region_name, "kmeans")
    role = create_role("kmeans-role")
    print(f"role: {role}")
    estimator = sagemaker.estimator.Estimator(
        container,
        role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        output_path=output_path,
        sagemaker_session=session,
        hyperparameters={
            "k": "3",
            "feature_dim": "4",
            "mini_batch_size": "500",
        },
    )
    return estimator


def main():
    bucket_name = "ipinsights-86589a88-8765-41e7-9019-86560161e6e2"
    create_bucket(bucket_name)

    training_path = "trianing.csv"
    with open("training.csv", "rb") as f:
        data = f.read().decode("utf-8")
    upload_data(bucket_name, training_path, data)

    job = create_training_artifact(f"s3://{bucket_name}/output")
    # Specify the data channels for training
    print(f"s3://{bucket_name}/{training_path}")
    # job.fit({"train": f"s3://{bucket_name}/{training_path}"})

    train_input = TrainingInput(
        f"s3://{bucket_name}/{training_path}", content_type="text/csv"
    )
    job.fit({"train": train_input})


if __name__ == "__main__":
    main()

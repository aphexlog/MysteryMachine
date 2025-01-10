import boto3
import json


def create_bucket(bucket: str) -> None:
    """Create an S3 bucket if it doesn't exist."""
    s3 = boto3.client("s3")
    try:
        s3.create_bucket(Bucket=bucket)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass


def upload_data(bucket: str, path: str, data: str) -> None:
    """Upload data to an S3 bucket at the specified path."""
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=path, Body=data)


def create_role(role_name: str) -> str:
    """Create or get an IAM role for SageMaker."""
    iam = boto3.client("iam")

    try:
        role = iam.get_role(RoleName=role_name)
    except iam.exceptions.NoSuchEntityException:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
        )

        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        )

    return role["Role"]["Arn"]

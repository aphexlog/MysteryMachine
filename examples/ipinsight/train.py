import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
import logging
from common.aws_utils import create_bucket, upload_data, create_role

logger = logging.getLogger()
session = sagemaker.Session()

# Create the IP Insights estimator
def create_training_artifact(output_path: str) -> sagemaker.estimator.Estimator:
    # Get the IP Insights container image
    container = get_image_uri(session.boto_region_name, "ipinsights")
    role = create_role("ipinsights-role")
    print(f"role: {role}")

    estimator = sagemaker.estimator.Estimator(
        container,
        role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        output_path=output_path,
        sagemaker_session=session,
        hyperparameters={
            "num_entity_vectors": "10000",
            "vector_dim": "128",
            "epochs": "10",
            "learning_rate": "0.001",
        },
    )
    return estimator

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

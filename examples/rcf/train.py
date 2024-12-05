from sagemaker.inputs import TrainingInput
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
import logging
from common.aws_utils import create_bucket, upload_data, create_role

logger = logging.getLogger()
session = sagemaker.Session()


def create_training_artifact(
    output_path: str, feature_dim: int, num_components: int
) -> sagemaker.estimator.Estimator:
    container = get_image_uri(session.boto_region_name, "randomcutforest")
    role = create_role("rcf-role")
    print(f"role: {role}")
    estimator = sagemaker.estimator.Estimator(
        container,
        role,
        instance_count=1,
        instance_type="ml.p2.xlarge",
        output_path=output_path,
        sagemaker_session=session,
        hyperparameters={
            "feature_dim": str(feature_dim),
        },
    )
    return estimator


def main():
    bucket_name = "rcf-86589a88-8765-41e7-9019-86560161e6e2"
    create_bucket(bucket_name)

    training_path = "data.csv"
    with open("data.csv", "rb") as f:
        data = f.read().decode("utf-8")
    upload_data(bucket_name, training_path, data)

    job = create_training_artifact(
        f"s3://{bucket_name}/output", feature_dim=33, num_components=3
    )
    # Specify the data channels for training
    print(f"s3://{bucket_name}/{training_path}")

    train_input = TrainingInput(
        f"s3://{bucket_name}/{training_path}", content_type="text/csv"
    )
    job.fit({"train": train_input})


if __name__ == "__main__":
    main()

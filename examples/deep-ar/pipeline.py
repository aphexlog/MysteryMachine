import sagemaker
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import ParameterString

# Define SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Parameters for pipeline
input_data_uri = ParameterString(
    name="InputDataUri", default_value="s3://your-bucket/path/to/time-series-data"
)

# Processing step
processor = ScriptProcessor(
    role=role,
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    instance_count=1,
    instance_type="ml.m5.large",
)

processing_step = ProcessingStep(
    name="TimeSeriesDataProcessing",
    processor=processor,
    inputs=[
        sagemaker.processing.ProcessingInput(  # FIX: This is broken
            source=input_data_uri, destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(  # FIX: This is broken
            output_name="processed_data", source="/opt/ml/processing/output"
        )
    ],
    code="scripts/processing.py",
)

# Training step using DeepAR
estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve(  # FIX: This is broken
        "forecasting-deepar", sagemaker_session.boto_region_name
    ),
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path="s3://your-bucket/path/to/output",
)

training_step = TrainingStep(
    name="TimeSeriesTraining",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[  # FIX: This is broken
                "processed_data"
            ].S3Output.S3Uri
        )
    },
)

# Creating the pipeline
pipeline = Pipeline(name="TimeSeriesPipeline", steps=[processing_step, training_step])

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start(
        parameters={"InputDataUri": "s3://your-bucket/path/to/time-series-data"}
    )
    execution.wait()

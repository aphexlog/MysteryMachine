import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

# Initialize a SageMaker session
session = sagemaker.Session()

# Get the execution role - this is used to give SageMaker access to your AWS resources
role = get_execution_role()

# Specify the data paths in S3
training_data = 's3://your-bucket/path/to/train/data'
validation_data = 's3://your-bucket/path/to/validation/data'

# Get the IP Insights container image
container = get_image_uri(session.boto_region_name, 'ipinsights')

# Create the IP Insights estimator
ip_insights = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.c5.xlarge',
    output_path='s3://your-bucket/output',
    sagemaker_session=session,
    # Hyperparameters for IP Insights
    hyperparameters={
        'num_entity_vectors': 10000,  # Number of entity embeddings
        'vector_dim': 128,            # Size of embeddings
        'batch_size': 1000,           # Training batch size
        'epochs': 10,                 # Number of training epochs
        'learning_rate': 0.001,       # Learning rate
        'num_ip_addresses': 10000,    # Maximum number of IP addresses to track
    }
)

# Specify the data channels for training
ip_insights.fit({
    'train': training_data,
    'validation': validation_data
})

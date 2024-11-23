# IP Insight Implementation Guide

This guide walks you through using the IP Insight implementation for anomaly detection in IP-based user behavior.

## Overview

IP Insight is an unsupervised learning algorithm that detects suspicious IP addresses by learning to accurately predict associations between IP addresses and entities (such as user names or account numbers).

## Files Description

- `process_csv.py`: Handles data preprocessing and transformation
- `run.py`: Contains the main execution logic including:
  - AWS bucket creation
  - Data upload functionality
  - IAM role management
  - SageMaker training configuration
- `training.csv`: Sample training dataset
- `validation.csv`: Sample validation dataset for model evaluation

## Usage

1. **Data Preparation**
   ```python
   from process_csv import process_data
   process_data('input.csv', 'processed.csv')
   ```

2. **Training Setup**
   ```python
   from run import create_bucket, upload_data, create_role
   
   # Create S3 bucket
   bucket_name = 'your-bucket-name'
   create_bucket(bucket_name)
   
   # Upload training data
   upload_data(bucket_name, 'path/to/data', 'training_data')
   
   # Setup IAM role
   role_arn = create_role('IPInsightRole')
   ```

3. **Model Training**
   ```python
   from run import create_training_artifact
   
   estimator = create_training_artifact('output/path')
   estimator.fit()
   ```

## Data Format

The training data should include:
- Entity IDs (e.g., usernames, account numbers)
- IP addresses
- Timestamps (optional)
- Additional features (optional)

## Best Practices

- Clean your input data thoroughly
- Use a representative validation dataset
- Monitor training metrics in SageMaker
- Start with a small dataset for testing

## Troubleshooting

Common issues and solutions:
- Ensure AWS credentials are properly configured
- Verify S3 bucket permissions
- Check data format matches expected schema

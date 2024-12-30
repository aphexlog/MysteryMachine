# SageMaker Examples

This repository provides practical examples of using Amazon SageMaker for different machine learning tasks. It's designed to help beginners understand how to:
- Preprocess data for SageMaker
- Set up and configure SageMaker training jobs
- Work with different SageMaker built-in algorithms

## Examples

1. **IP Insight** - Anomaly detection for IP addresses
   - Learn how to process CSV data
   - Use SageMaker's IP Insight algorithm
   - Handle training and validation datasets

2. **K-Means Clustering** - Unsupervised learning example
   - Understand clustering with SageMaker
   - Configure k-means hyperparameters
   - Process numerical data for clustering

3. **PCA (Principal Component Analysis)** - Dimensionality reduction
   - Learn about feature reduction
   - Configure PCA parameters
   - Handle high-dimensional data

4. **Random Cut Forest (RCF)** - Unsupervised anomaly detection
   - Learn about anomaly detection
   - Configure RCF parameters
   - Process time-series data

## Getting Started

1. Install dependencies from the root directory:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   ```

3. Choose an example from the `examples/` directory and follow its README.

## Project Structure

- `common/` - Shared utilities for AWS and SageMaker
- `examples/` - Individual algorithm examples
- Each example contains:
  - README.md with specific instructions
  - Data preprocessing code (if needed)
  - Training script

## Prerequisites

- AWS Account
- Python 3.7+
- Basic understanding of machine learning concepts
- AWS CLI configured with appropriate permissions

## License

See the [LICENSE](LICENSE) file for details.

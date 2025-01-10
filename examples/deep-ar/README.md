# SageMaker Time Series Pipeline with DeepAR

## Steps to Run the Pipeline

1. **Upload your dataset to S3**.
2. **Modify the S3 bucket paths in `pipeline.py`**.
3. **Run the pipeline script**:
   ```bash
   python DeepAR/pipeline.py
   ```
4. The pipeline will automatically process the data and train a DeepAR model.

Ensure your IAM role has the necessary permissions to access SageMaker and S3.

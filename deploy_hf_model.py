import os
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from botocore.exceptions import ClientError

def main():
    try:
        # Get SageMaker session and region
        sagemaker_session = sagemaker.Session()
        region = sagemaker_session.boto_region_name
        
        # Use your SageMaker execution role ARN
        role = "Add your Role"  # Replace with your role ARN

        # Get the S3 bucket name from environment or use default
        bucket_name = os.getenv('SAGEMAKER_BUCKET', 'deepseek-models')
        model_s3_uri = f"s3://{bucket_name}/model.tar.gz"

        # Upload model to S3 if not already there
        if not os.path.exists('model.tar.gz'):
            raise FileNotFoundError("model.tar.gz not found in current directory")

        # Upload to S3
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file('model.tar.gz', bucket_name, 'model.tar.gz')
        except ClientError as e:
            print(f"Error uploading to S3: {e}")
            return

        # Hugging Face Model config
        hf_model = HuggingFaceModel(
            transformers_version="4.37.0",
            pytorch_version="2.1.1",
            py_version="py310",
            model_data=model_s3_uri,
            role=role,
            env={
                "HF_MODEL_ID": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "HF_TASK": "text-generation"
            }
        )

        # Deploy to an endpoint
        print("Deploying model to SageMaker endpoint...")
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type="ml.g4dn.xlarge",  # GPU instance for better performance
            endpoint_name="deepseek-endpoint"
        )
        print(f"Model deployed successfully to endpoint: {predictor.endpoint_name}")

    except Exception as e:
        print(f"Error during deployment: {e}")

if __name__ == "__main__":
    main()

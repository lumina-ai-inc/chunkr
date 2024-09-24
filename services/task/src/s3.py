import boto3
from urllib.parse import urlparse

from src.configs.aws_config import AWS__ACCESS_KEY, AWS__SECRET_KEY, AWS__ENDPOINT, AWS__REGION

def upload_file_to_s3(local_file_path, s3_path):
    """
    Upload a file to S3 given an S3 path starting with 's3://'.

    :param local_file_path: Path to the local file to be uploaded
    :param s3_path: S3 path in the format 's3://bucket-name/object-key'
    """
    parsed_url = urlparse(s3_path)
    if parsed_url.scheme != 's3':
        raise ValueError("Invalid S3 path. Must start with 's3://'")

    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')

    s3_client = boto3.client('s3',
                             aws_access_key_id=AWS__ACCESS_KEY,
                             aws_secret_access_key=AWS__SECRET_KEY,
                             endpoint_url=AWS__ENDPOINT,
                             region_name=AWS__REGION)

    try:
        s3_client.upload_file(local_file_path, bucket_name, object_key)
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        raise

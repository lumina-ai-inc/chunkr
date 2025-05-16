import os
import boto3
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from tempfile import _TemporaryFileWrapper
import tempfile
import dotenv

dotenv.load_dotenv(override=True)

class FileType(str, Enum):
    FINANCIAL = "financial"
    BILLING = "billing"
    TAX = "tax"
    SUPPLY_CHAIN = "supply_chain"
    TECHNICAL = "technical"
    RESEARCH = "research"
    LEGAL = "legal"
    GOVERNMENT = "government"
    CONSULTING = "consulting"
    MAGAZINE = "magazine"
    NEWSPAPER = "newspaper"
    TEXTBOOK = "textbook"
    HISTORICAL = "historical"
    PATENT = "patent"
    EDUCATION = "education"
    MEDICAL = "medical"
    REAL_ESTATE = "real_estate"
    MISCELLANEOUS = "miscellaneous"
    TEST = "test"
    # add more here

class S3Config(BaseModel):
    access_key: str = Field(default_factory=lambda: os.environ.get("AWS__ACCESS_KEY") or "")
    secret_key: str = Field(default_factory=lambda: os.environ.get("AWS__SECRET_KEY") or "")
    endpoint: str = Field(default_factory=lambda: os.environ.get("AWS__ENDPOINT") or "https://s3.us-west-1.amazonaws.com")
    presigned_url_endpoint: str = Field(default_factory=lambda: os.environ.get("AWS__PRESIGNED_URL_ENDPOINT") or "")
    bucket_name: str = "chunkr-datasets"

def upload_to_s3(temp_file: _TemporaryFileWrapper, file_type: FileType, file_name: Optional[str] = None) -> str:
    config = S3Config()
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        endpoint_url=config.endpoint
    )
    
    if file_name is None:
        file_name = os.path.basename(temp_file.name)
    
    object_key = f"dla-dataset/{file_type.value}/{file_name}"
    
    # Ensure the bucket exists
    try:
        s3_client.head_bucket(Bucket=config.bucket_name)
    except s3_client.exceptions.NoSuchBucket:
        try:
            s3_client.create_bucket(Bucket=config.bucket_name)
        except Exception as e:
            raise RuntimeError(f"Failed to create bucket: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to access bucket: {e}")
    
    temp_file.seek(0)
    s3_client.upload_fileobj(
        temp_file,
        config.bucket_name,
        object_key
    )
    
    return f"{config.presigned_url_endpoint}{config.bucket_name}/{object_key}"

def get_presigned_url(file_type: FileType, file_name: str, expiration: int = 3600) -> str:
    config = S3Config()
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        endpoint_url=config.endpoint
    )
    
    object_key = f"{file_type.value}/{file_name}"
    
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': config.bucket_name,
            'Key': object_key
        },
        ExpiresIn=expiration
    )
    
    return url
def test_upload():
    with open("test.pdf", "rb") as f:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(f.read())
            temp_file.flush()  # Ensure all data is written to disk
            
            # Pass the file to upload_to_s3 while it's still open
            url = upload_to_s3(temp_file, FileType.TEST, "test.pdf")
            print(f"Uploaded test.pdf to: {url}")
        
        os.unlink(temp_file.name)
        
# if __name__ == "__main__":
#     test_upload() #uncomment this to test


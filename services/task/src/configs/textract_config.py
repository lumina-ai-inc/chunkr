import boto3
import dotenv
import os
import subprocess

dotenv.load_dotenv(override=True)

TEXTRACT__AWS_ACCESS_KEY = os.getenv("TEXTRACT__AWS_ACCESS_KEY")
TEXTRACT__AWS_SECRET_KEY = os.getenv("TEXTRACT__AWS_SECRET_KEY")
TEXTRACT__AWS_REGION = os.getenv("TEXTRACT__AWS_REGION")

def login_aws():
    TEXTRACT__AWS_ACCESS_KEY = os.getenv("TEXTRACT__AWS_ACCESS_KEY")
    TEXTRACT__AWS_SECRET_KEY = os.getenv("TEXTRACT__AWS_SECRET_KEY")
    TEXTRACT__AWS_REGION = os.getenv("TEXTRACT__AWS_REGION")
    print("values:", TEXTRACT__AWS_ACCESS_KEY, TEXTRACT__AWS_SECRET_KEY, TEXTRACT__AWS_REGION)
    boto3.setup_default_session(aws_access_key_id=TEXTRACT__AWS_ACCESS_KEY, aws_secret_access_key=TEXTRACT__AWS_SECRET_KEY, region_name=TEXTRACT__AWS_REGION)

    # if TEXTRACT__AWS_ACCESS_KEY:
    #     try:
    #         subprocess.run(["aws", "configure", "set", "aws_access_key_id", TEXTRACT__AWS_ACCESS_KEY], check=True)
    #         subprocess.run(["aws", "configure", "set", "aws_secret_access_key", TEXTRACT__AWS_SECRET_KEY], check=True)
    #         subprocess.run(["aws", "configure", "set", "default.region", TEXTRACT__AWS_REGION], check=True)
    #         print("AWS login successful for TEXTRACT-specific credentials.")
    #     except subprocess.CalledProcessError:
    #         print("Error: AWS login failed for TEXTRACT-specific credentials.")
    # else:
    #     print("No TEXTRACT-specific AWS credentials found.")
    

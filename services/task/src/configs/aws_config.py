import dotenv
import os
import subprocess

dotenv.load_dotenv(override=True)

AWS__ENDPOINT = os.getenv("AWS__ENDPOINT")
AWS__ACCESS_KEY = os.getenv("AWS__ACCESS_KEY")
AWS__SECRET_KEY = os.getenv("AWS__SECRET_KEY")
AWS__REGION = os.getenv("AWS__REGION")


TASK__AWS_ACCESS_KEY = os.getenv("TASK__AWS_ACCESS_KEY")
TASK__AWS_SECRET_KEY = os.getenv("TASK__AWS_SECRET_KEY")
TASK__AWS_REGION = os.getenv("TASK__AWS_REGION")

def login_aws():
    TASK__AWS_ACCESS_KEY = os.getenv("TASK__AWS_ACCESS_KEY")
    TASK__AWS_SECRET_KEY = os.getenv("TASK__AWS_SECRET_KEY")
    TASK__AWS_REGION = os.getenv("TASK__AWS_REGION")
    if TASK__AWS_ACCESS_KEY:
        try:
            subprocess.run(["aws", "configure", "set", "aws_access_key_id", TASK__AWS_ACCESS_KEY], check=True)
            subprocess.run(["aws", "configure", "set", "aws_secret_access_key", TASK__AWS_SECRET_KEY], check=True)
            subprocess.run(["aws", "configure", "set", "default.region", TASK__AWS_REGION], check=True)
            print("AWS login successful for task-specific credentials.")
        except subprocess.CalledProcessError:
            print("Error: AWS login failed for task-specific credentials.")
    else:
        print("No task-specific AWS credentials found.")

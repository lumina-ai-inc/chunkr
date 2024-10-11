import dotenv
import os

dotenv.load_dotenv(override=True)

AWS__ENDPOINT = os.getenv("AWS__ENDPOINT")
AWS__ACCESS_KEY = os.getenv("AWS__ACCESS_KEY")
AWS__SECRET_KEY = os.getenv("AWS__SECRET_KEY")
AWS__REGION = os.getenv("AWS__REGION")


TASK__AWS__ACCESS_KEY = os.getenv("TASK__AWS__ACCESS_KEY")
TASK__AWS__SECRET_KEY = os.getenv("TASK__AWS__SECRET_KEY")
TASK__AWS__REGION = os.getenv("TASK__AWS__REGION")
if TASK__AWS__ACCESS_KEY:
    import subprocess

    login_script = "login.sh"
    try:
        subprocess.run([login_script, TASK__AWS__ACCESS_KEY, TASK__AWS__SECRET_KEY, TASK__AWS__REGION], check=True)
        print("AWS login successful for task-specific credentials.")
    except subprocess.CalledProcessError:
        print("Error: AWS login failed for task-specific credentials.")
    except FileNotFoundError:
        print(f"Error: Login script not found at {login_script}")
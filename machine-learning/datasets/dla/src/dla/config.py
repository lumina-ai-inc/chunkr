from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # GCS via S3-compatible API
    google_access_key: str = Field(..., validation_alias="GOOGLE_ACCESS_KEY")
    google_secret_key: str = Field(..., validation_alias="GOOGLE_SECRET_KEY")
    google_endpoint: str = Field(..., validation_alias="GOOGLE_ENDPOINT")
    bucket_name: str = Field(..., validation_alias="GOOGLE_BUCKET_NAME")

    # download_dir: str = "data/raw" # No longer needed

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }
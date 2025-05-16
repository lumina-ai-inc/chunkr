import os, hashlib, uuid
from pathlib import Path
import boto3
from .config import Settings

def get_s3_client(settings: Settings):
    return boto3.client(
        "s3",
        aws_access_key_id=settings.google_access_key,
        aws_secret_access_key=settings.google_secret_key,
        endpoint_url=settings.google_endpoint,
    )

def list_page_keys(client, bucket_name):
    """Yields all object keys ending in .jpg/.png under any path."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".jpg", ".jpeg", ".png")):
                yield key

def parse_key(key: str):
    """
    Expect: {user_id}/{doc_id}/images/pages/page_{i}.jpg/.png/.jpeg
    Returns (user_id, doc_id, page_index, filename) or None on mismatch.
    """
    parts = key.split("/")
    # Expecting 4 parts: user_id, doc_id, images, pages, filename
    if len(parts) != 5:
        return None

    try:
        user_id, doc_id, images_folder, pages_folder, filename = parts
        # Validate folder names
        if images_folder != "images" or pages_folder != "pages":
            return None
        # Validate filename format and extract index
        if not filename.startswith("page_") or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
             return None
        idx_str = filename.split("_")[1].split(".")[0]
        idx = int(idx_str)
        return user_id, doc_id, idx, filename
    except (ValueError, IndexError): # Catch potential errors during splitting/conversion
        return None

def download_and_hash(client, bucket_name, key: str):
    """Downloads object bytes into memory, computes SHA256 hash."""
    try:
        resp = client.get_object(Bucket=bucket_name, Key=key)
        data = resp["Body"].read()
        h = hashlib.sha256(data).hexdigest()
    except Exception as e:
        print(f"  Error downloading/hashing key {key}: {e}")
        return None, None # Return None if download/hash fails

    # return metadata needed for DB insert
    page_id = str(uuid.uuid4())
    return page_id, h
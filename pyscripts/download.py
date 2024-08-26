import requests
import os
import re


def download_file(url: str, output_path: str) -> str:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Extract filename from Content-Disposition header or URL
    content_disposition = response.headers.get("Content-Disposition")
    if content_disposition:
        filename = re.findall("filename=(.+)", content_disposition)[0].strip('"')
    else:
        filename = os.path.basename(url.split("?")[0])

    # Check if output_path is a directory or a file path
    if os.path.isdir(output_path):
        file_path = os.path.join(output_path, filename)
    else:
        file_path = output_path

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return file_path

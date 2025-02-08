from .configuration import Configuration
import base64
import httpx
import io
from pathlib import Path
from PIL import Image
from typing import Union, Tuple, BinaryIO, Optional

async def prepare_file(file: Union[str, Path, BinaryIO, Image.Image], client: httpx.AsyncClient = None) -> Tuple[str, BinaryIO]:
    """Convert various file types into a tuple of (filename, file-like object).

    Args:
        file: Input file, can be:
            - URL string starting with http:// or https://
            - Base64 string
            - Local file path (will be converted to base64)
            - Opened binary file (will be converted to base64)
            - PIL/Pillow Image object (will be converted to base64)

    Returns:
        Tuple[str, BinaryIO]: (filename, file-like object) ready for upload

    Raises:
        FileNotFoundError: If the file path doesn't exist
        TypeError: If the file type is not supported
        ValueError: If the URL is invalid or unreachable
        ValueError: If the MIME type is unsupported
    """
    if isinstance(file, str) and (
        file.startswith("http://") or file.startswith("https://")
    ):
        if not client:
            raise ValueError("Client must be provided to validate URLs")
        response = await client.head(file)
        response.raise_for_status()
        return None, None

    # Handle base64 strings
    if isinstance(file, str) and "," in file and ";base64," in file:
        try:
            header, base64_data = file.split(",", 1)
            base64.b64decode(base64_data)
            mime_type = header.split(":")[-1].split(";")[0].lower()
            mime_to_ext = {
                "application/pdf": "pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                "application/msword": "doc",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
                "application/vnd.ms-powerpoint": "ppt",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                "application/vnd.ms-excel": "xls",
                "image/jpeg": "jpg",
                "image/png": "png",
                "image/jpg": "jpg",
            }
            if mime_type not in mime_to_ext:
                raise ValueError(f"Unsupported MIME type: {mime_type}")

            format = mime_to_ext[mime_type]
            return None, file

        except Exception as e:
            raise ValueError(f"Invalid base64 string: {str(e)}")

    # Handle file paths - convert to base64
    if isinstance(file, (str, Path)):
        path = Path(file).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file}")
        
        with open(path, "rb") as f:
            file_content = f.read()
            file_ext = path.suffix.lower().lstrip('.')
            if not file_ext:
                raise ValueError("File must have an extension")
            base64_str = base64.b64encode(file_content).decode()
            return path.name, base64_str

    # Handle PIL Images - convert to base64
    if isinstance(file, Image.Image):
        img_byte_arr = io.BytesIO()
        format = file.format or "PNG"
        file.save(img_byte_arr, format=format)
        img_byte_arr.seek(0)
        base64_str = base64.b64encode(img_byte_arr.getvalue()).decode()
        return None, base64_str

    # Handle file-like objects - convert to base64
    if hasattr(file, "read") and hasattr(file, "seek"):
        file.seek(0)
        file_content = file.read()
        name = getattr(file, "name", "document")
        file_ext = Path(name).suffix.lower().lstrip('.')
        if not file_ext:
            raise ValueError("File must have an extension")
        base64_str = base64.b64encode(file_content).decode()
        return Path(name).name, base64_str

    raise TypeError(f"Unsupported file type: {type(file)}")


async def prepare_upload_data(
    file: Optional[Union[str, Path, BinaryIO, Image.Image]] = None,
    config: Optional[Configuration] = None,
    client: httpx.AsyncClient = None,
) -> dict:
    """Prepare data dictionary for upload.

    Args:
        file: The file to upload
        config: Optional configuration settings
        client: HTTP client for downloading remote files

    Returns:
        dict: JSON-serializable data dictionary ready for upload
    """
    data = {}
    if file:
        if isinstance(file, str) and (
            file.startswith("http://") or file.startswith("https://")
        ):
            data["file"] = file
        else:
            filename, base64_str = await prepare_file(file, client)
            data["file"] = base64_str
            data["file_name"] = filename

    if config:
        data.update(config.model_dump(mode="json", exclude_none=True))
        
    return data

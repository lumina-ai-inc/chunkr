import pytest
from pathlib import Path
from PIL import Image
import base64
import io
import tempfile

from chunkr_ai import Chunkr

@pytest.fixture
def sample_path():
    return Path("tests/files/test.pdf")

@pytest.fixture
def sample_url():
    return "https://chunkr-web.s3.us-east-1.amazonaws.com/landing_page/input/science.pdf"

@pytest.fixture
def sample_image():
    return Image.open("tests/files/test.jpg")

@pytest.fixture
def client():
    client = Chunkr()
    yield client

@pytest.mark.asyncio
async def test_send_file_path(client, sample_path):
    response = await client.upload(sample_path)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_file_path_str(client, sample_path):
    response = await client.upload(str(sample_path))
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_file_relative_path_str(client):
    response = await client.upload("./tests/files/test.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_file_url(client, sample_url):
    response = await client.upload(sample_url)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_opened_file(client, sample_path):
    with open(sample_path, "rb") as f:
        response = await client.upload(f)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_pil_image(client, sample_image):
    response = await client.upload(sample_image)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_base64_file(client, sample_path):
    # Read file and convert to base64
    with open(sample_path, "rb") as f:
        base64_content = base64.b64encode(f.read())
    response = await client.upload(base64_content)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_base64_file_w_decode(client, sample_path):
    # Read file and convert to base64
    with open(sample_path, "rb") as f:
        base64_content = base64.b64encode(f.read()).decode()
    response = await client.upload(base64_content)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_base64_file_with_data_url(client, sample_path):
    with open(sample_path, "rb") as f:
        base64_content = base64.b64encode(f.read()).decode('utf-8')
    response = await client.upload(f"data:application/pdf;base64,{base64_content}")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_base64_file_with_filename(client, sample_path):
    # Read file and convert to base64
    with open(sample_path, "rb") as f:
        base64_content = base64.b64encode(f.read()).decode('utf-8')
    
    response = await client.upload(base64_content, filename="test.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_file_like_no_name_attribute(client, sample_path):
    # Create a file-like object without a name attribute
    class NamelessBuffer:
        def __init__(self, content):
            self.buffer = io.BytesIO(content)
        
        def read(self):
            return self.buffer.read()
        
        def seek(self, pos):
            return self.buffer.seek(pos)
    
    with open(sample_path, "rb") as f:
        content = f.read()
    
    nameless_buffer = NamelessBuffer(content)
    response = await client.upload(nameless_buffer, filename="test.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_file_like_none_name(client, sample_path):
    # Create a file-like object with None as name
    class NoneNameBuffer:
        def __init__(self, content):
            self.buffer = io.BytesIO(content)
            self.name = None
        
        def read(self):
            return self.buffer.read()
        
        def seek(self, pos):
            return self.buffer.seek(pos)
    
    with open(sample_path, "rb") as f:
        content = f.read()
    
    none_name_buffer = NoneNameBuffer(content)
    response = await client.upload(none_name_buffer, filename="test.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_file_like_no_extension(client, sample_path):
    # Create a file-like object with a name but no extension
    class NoExtensionBuffer:
        def __init__(self, content):
            self.buffer = io.BytesIO(content)
            self.name = "test_document"
        
        def read(self):
            return self.buffer.read()
        
        def seek(self, pos):
            return self.buffer.seek(pos)
    
    with open(sample_path, "rb") as f:
        content = f.read()
    
    no_ext_buffer = NoExtensionBuffer(content)
    response = await client.upload(no_ext_buffer, filename="test.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_spooled_temporary_file(client, sample_path):
    # Test with SpooledTemporaryFile which is what the user is using
    with open(sample_path, "rb") as f:
        content = f.read()
    
    temp_file = tempfile.SpooledTemporaryFile()
    temp_file.write(content)
    temp_file.seek(0)
    
    response = await client.upload(temp_file, filename="test.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_bytearray(client, sample_path):
    # Read file and convert to bytearray
    with open(sample_path, "rb") as f:
        content = bytearray(f.read())
    
    response = await client.upload(content, filename="test.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_memoryview(client, sample_path):
    # Read file and convert to memoryview
    with open(sample_path, "rb") as f:
        content_bytes = f.read()
        content = memoryview(content_bytes)
    
    response = await client.upload(content, filename="test.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_with_explicit_filename_pdf(client, sample_path):
    response = await client.upload(sample_path, filename="custom_name.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_with_explicit_filename_image(client, sample_image):
    response = await client.upload(sample_image, filename="custom_image.jpg")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_with_special_character_filename(client, sample_path):
    response = await client.upload(sample_path, filename="test file (1)&%$#@!.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_filename_with_non_matching_extension(client, sample_path):
    # Test providing a filename with a different extension than the actual file
    response = await client.upload(sample_path, filename="document.docx")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_bytes_with_explicit_filename(client, sample_path):
    with open(sample_path, "rb") as f:
        content = f.read()
    
    # For bytes objects, filename is required to know the file type
    response = await client.upload(content, filename="document.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_bytearray_with_explicit_filename(client, sample_path):
    with open(sample_path, "rb") as f:
        content = bytearray(f.read())
    
    response = await client.upload(content, filename="document.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_memoryview_with_explicit_filename(client, sample_path):
    with open(sample_path, "rb") as f:
        content_bytes = f.read()
        content = memoryview(content_bytes)
    
    response = await client.upload(content, filename="document.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_unicode_filename(client, sample_path):
    # Test with a filename containing Unicode characters
    response = await client.upload(sample_path, filename="测试文件.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_very_long_filename(client, sample_path):
    # Test with an extremely long filename
    long_name = "a" * 200 + ".pdf"  # 200 character filename
    response = await client.upload(sample_path, filename=long_name)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_filename_without_extension(client, sample_path):
    # Test with a filename that has no extension
    with open(sample_path, "rb") as f:
        content = f.read()
    
    # This test verifies that the system uses the provided filename even without extension
    response = await client.upload(content, filename="document_without_extension")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_custom_file_like_with_filename(client, sample_path):
    # A more complex file-like object implementation
    class CustomFileWrapper:
        def __init__(self, content):
            self.buffer = io.BytesIO(content)
            self.position = 0
            self.name = "original_name.txt"  # Should be overridden by explicit filename
        
        def read(self, size=-1):
            return self.buffer.read(size)
        
        def seek(self, position, whence=0):
            return self.buffer.seek(position, whence)
        
        def tell(self):
            return self.buffer.tell()
        
        def close(self):
            self.buffer.close()
    
    with open(sample_path, "rb") as f:
        content = f.read()
    
    custom_file = CustomFileWrapper(content)
    response = await client.upload(custom_file, filename="custom_wrapper.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_seek_at_nonzero_position(client, sample_path):
    # Test with a file-like object that's not at position 0
    with open(sample_path, "rb") as f:
        content = f.read()
    
    buffer = io.BytesIO(content)
    buffer.seek(100)  # Move position to 100
    
    response = await client.upload(buffer, filename="seek_test.pdf")
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_reused_file_object(client, sample_path):
    # Test that a file object can be reused after being processed
    with open(sample_path, "rb") as f:
        response1 = await client.upload(f, filename="first_use.pdf")
        f.seek(0)  # Reset position
        response2 = await client.upload(f, filename="second_use.pdf")
    
    assert response1.task_id is not None
    assert response1.status == "Succeeded"
    assert response2.task_id is not None
    assert response2.status == "Succeeded"

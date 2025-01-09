import pytest
import os
from pathlib import Path
from PIL import Image
import io
from chunkr_ai import Chunkr, ChunkrAsync
from chunkr_ai.api.models import TaskResponse

# Test fixtures
@pytest.fixture
def chunkr():
    return Chunkr()

@pytest.fixture
def async_chunkr():
    return ChunkrAsync()

@pytest.fixture
def sample_pdf():
    # Create a temporary PDF file for testing
    content = b"%PDF-1.4 test content"
    pdf_path = Path("test_document.pdf")
    pdf_path.write_bytes(content)
    yield str(pdf_path)
    pdf_path.unlink()  # Cleanup after tests

@pytest.fixture
def sample_image():
    # Create a test image
    img = Image.new('RGB', (100, 100), color='red')
    return img

def test_prepare_file_string(chunkr, sample_pdf):
    filename, file_obj = chunkr._prepare_file(sample_pdf)
    assert filename == "test_document.pdf"
    assert hasattr(file_obj, 'read')

def test_prepare_file_image(chunkr, sample_image):
    filename, file_obj = chunkr._prepare_file(sample_image)
    assert filename == "image.png"
    assert isinstance(file_obj, io.BytesIO)

def test_prepare_file_bytes(chunkr):
    test_bytes = b"test content"
    filename, file_obj = chunkr._prepare_file(test_bytes)
    assert filename == "document"
    assert isinstance(file_obj, io.BytesIO)

def test_send_file_string(chunkr, sample_pdf):
    response = chunkr.upload(sample_pdf)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status in ["pending", "processing", "completed"]

def test_send_file_image(chunkr, sample_image):
    response = chunkr.upload(sample_image)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status in ["pending", "processing", "completed"]

def test_send_file_bytes(chunkr):
    test_bytes = b"This is a test document content"
    response = chunkr.upload(test_bytes)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status in ["pending", "processing", "completed"]
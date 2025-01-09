import pytest
from pathlib import Path
from PIL import Image

from chunkr_ai import Chunkr, ChunkrAsync
from chunkr_ai.models import Configuration, OcrStrategy, TaskResponse

@pytest.fixture
def chunkr():
    return Chunkr()

@pytest.fixture
def async_chunkr():
    return ChunkrAsync()

@pytest.fixture
def sample_path():
    return Path("tests/files/test.pdf")
    
@pytest.fixture
def sample_image():
    img = Image.open("tests/files/test.jpg")
    return img

def test_send_file_path(chunkr, sample_path):
    response = chunkr.upload(sample_path)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

def test_send_file_path_str(chunkr, sample_path):
    response = chunkr.upload(str(sample_path))
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

def test_send_opened_file(chunkr, sample_path):
    with open(sample_path, 'rb') as f:
        response = chunkr.upload(f)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
# def test_send_pil_image(chunkr, sample_image):
#     response = chunkr.upload(sample_image)
    
#     assert isinstance(response, TaskResponse)
#     assert response.task_id is not None
#     assert response.status == "Succeeded"

def test_ocr_auto(chunkr, sample_path):
    response = chunkr.upload(sample_path, Configuration(
        ocr_strategy=OcrStrategy.AUTO
    ))
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

def test_ocr_expires_in(chunkr, sample_path):
    response = chunkr.upload(sample_path, Configuration(
        expires_in=10
    ))
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
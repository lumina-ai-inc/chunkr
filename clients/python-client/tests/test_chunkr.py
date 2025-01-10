import pytest
from pathlib import Path
from PIL import Image

from chunkr_ai import Chunkr, ChunkrAsync
from chunkr_ai.models import (
    ChunkProcessing, 
    Configuration, 
    GenerationStrategy, 
    GenerationConfig,
    JsonSchema,
    OcrStrategy, 
    Property,
    SegmentationStrategy, 
    SegmentProcessing, 
    TaskResponse, 
)

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
    
def test_send_pil_image(chunkr, sample_image):
    response = chunkr.upload(sample_image)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"

def test_ocr_auto(chunkr, sample_path):
    response = chunkr.upload(sample_path, Configuration(
        ocr_strategy=OcrStrategy.AUTO
    ))
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

def test_expires_in(chunkr, sample_path):
    response = chunkr.upload(sample_path, Configuration(
        expires_in=10
    ))
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
def test_chunk_processing(chunkr, sample_path):
    response = chunkr.upload(sample_path, Configuration(
        chunk_processing=ChunkProcessing(
            target_length=1024
        )
    ))
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
def test_segmentation_strategy_page(chunkr, sample_path):
    response = chunkr.upload(sample_path, Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE
    ))
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
def test_page_llm_html(chunkr, sample_path):
    response = chunkr.upload(sample_path, Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
        segment_processing=SegmentProcessing(
            page=GenerationConfig(
                html=GenerationStrategy.LLM
            )
        )
    ))
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

def test_page_llm(chunkr, sample_path):
    response = chunkr.upload(sample_path, Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
        segment_processing=SegmentProcessing(
            page=GenerationConfig(
                html=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM
            )
        )
    ))
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
  
def test_json_schema(chunkr, sample_path):
    response = chunkr.upload(sample_path, Configuration(
        json_schema=JsonSchema(
            title="Sales Data",
            properties=[
                Property(name="Person with highest sales", prop_type="string", description="The person with the highest sales"),
                Property(name="Person with lowest sales", prop_type="string", description="The person with the lowest sales"),
            ]
        )
    ))
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

async def test_async_send_file_path(async_chunkr, sample_path):
    response = await async_chunkr.upload(sample_path)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
    
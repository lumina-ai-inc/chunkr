import pytest
import pytest_asyncio
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

@pytest.fixture(params=[
    pytest.param(("sync", Chunkr()), id="sync"),
    pytest.param(("async", ChunkrAsync()), id="async")
])
def chunkr_client(request):
    return request.param

@pytest.fixture
def sample_path():
    return Path("tests/files/test.pdf")
    
@pytest.fixture
def sample_image():
    img = Image.open("tests/files/test.jpg")
    return img

@pytest.mark.asyncio
async def test_send_file_path(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = await client.upload(sample_path) if client_type == "async" else client.upload(sample_path)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_file_path_str(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = await client.upload(str(sample_path)) if client_type == "async" else client.upload(str(sample_path))
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_send_opened_file(chunkr_client, sample_path):
    client_type, client = chunkr_client
    with open(sample_path, 'rb') as f:
        response = await client.upload(f) if client_type == "async" else client.upload(f)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
@pytest.mark.asyncio
async def test_send_pil_image(chunkr_client, sample_image):
    client_type, client = chunkr_client
    response = await client.upload(sample_image) if client_type == "async" else client.upload(sample_image)
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"

@pytest.mark.asyncio
async def test_ocr_auto(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = await client.upload(sample_path, Configuration(
        ocr_strategy=OcrStrategy.AUTO
    )) if client_type == "async" else client.upload(sample_path, Configuration(
        ocr_strategy=OcrStrategy.AUTO
    ))
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_expires_in(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = await client.upload(sample_path, Configuration(
        expires_in=10
    )) if client_type == "async" else client.upload(sample_path, Configuration(
        expires_in=10
    ))
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
@pytest.mark.asyncio
async def test_chunk_processing(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = await client.upload(sample_path, Configuration(
        chunk_processing=ChunkProcessing(
            target_length=1024
        )
    )) if client_type == "async" else client.upload(sample_path, Configuration(
        chunk_processing=ChunkProcessing(
            target_length=1024
        )
    ))
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
@pytest.mark.asyncio
async def test_segmentation_strategy_page(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = await client.upload(sample_path, Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE
    )) if client_type == "async" else client.upload(sample_path, Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE
    ))
    
    assert isinstance(response, TaskResponse)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    
@pytest.mark.asyncio
async def test_page_llm_html(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = await client.upload(sample_path, Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
        segment_processing=SegmentProcessing(
            page=GenerationConfig(
                html=GenerationStrategy.LLM
            )
        )
    )) if client_type == "async" else client.upload(sample_path, Configuration(
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

@pytest.mark.asyncio
async def test_page_llm(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = await client.upload(sample_path, Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
        segment_processing=SegmentProcessing(
            page=GenerationConfig(
                html=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM
            )
        )
    )) if client_type == "async" else client.upload(sample_path, Configuration(
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
  
@pytest.mark.asyncio
async def test_json_schema(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = await client.upload(sample_path, Configuration(
        json_schema=JsonSchema(
            title="Sales Data",
            properties=[
                Property(name="Person with highest sales", prop_type="string", description="The person with the highest sales"),
                Property(name="Person with lowest sales", prop_type="string", description="The person with the lowest sales"),
            ]
        )
    )) if client_type == "async" else client.upload(sample_path, Configuration(
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
    
    
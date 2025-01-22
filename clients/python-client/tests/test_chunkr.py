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
    PipelineType,
    Property,
    SegmentationStrategy,
    SegmentProcessing,
    TaskResponse,
)


@pytest.fixture(
    params=[
        pytest.param(("sync", Chunkr()), id="sync"),
        pytest.param(("async", ChunkrAsync()), id="async"),
    ]
)
def chunkr_client(request):
    return request.param


@pytest.fixture
def sample_path():
    return Path("tests/files/test.pdf")


@pytest.fixture
def sample_image():
    img = Image.open("tests/files/test.jpg")
    return img


@pytest.fixture(params=[
    pytest.param(None, id="none_pipeline"),
    pytest.param(PipelineType.AZURE, id="azure_pipeline"),
])
def pipeline_type(request):
    return request.param


@pytest.mark.asyncio
async def test_send_file_path(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.upload(sample_path)
        if client_type == "async"
        else client.upload(sample_path)
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None


@pytest.mark.asyncio
async def test_send_file_path_str(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.upload(str(sample_path))
        if client_type == "async"
        else client.upload(str(sample_path))
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None


@pytest.mark.asyncio
async def test_send_opened_file(chunkr_client, sample_path):
    client_type, client = chunkr_client
    with open(sample_path, "rb") as f:
        response = (
            await client.upload(f) if client_type == "async" else client.upload(f)
        )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None


@pytest.mark.asyncio
async def test_send_pil_image(chunkr_client, sample_image):
    client_type, client = chunkr_client
    response = (
        await client.upload(sample_image)
        if client_type == "async"
        else client.upload(sample_image)
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"


@pytest.mark.asyncio
async def test_ocr_auto(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.upload(sample_path, Configuration(ocr_strategy=OcrStrategy.AUTO))
        if client_type == "async"
        else client.upload(sample_path, Configuration(ocr_strategy=OcrStrategy.AUTO))
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None


@pytest.mark.asyncio
async def test_expires_in(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.upload(sample_path, Configuration(expires_in=10))
        if client_type == "async"
        else client.upload(sample_path, Configuration(expires_in=10))
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None


@pytest.mark.asyncio
async def test_chunk_processing(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.upload(
            sample_path,
            Configuration(chunk_processing=ChunkProcessing(target_length=1024)),
        )
        if client_type == "async"
        else client.upload(
            sample_path,
            Configuration(chunk_processing=ChunkProcessing(target_length=1024)),
        )
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None


@pytest.mark.asyncio
async def test_segmentation_strategy_page(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.upload(
            sample_path, Configuration(segmentation_strategy=SegmentationStrategy.PAGE)
        )
        if client_type == "async"
        else client.upload(
            sample_path, Configuration(segmentation_strategy=SegmentationStrategy.PAGE)
        )
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None


@pytest.mark.asyncio
async def test_page_llm_html(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.upload(
            sample_path,
            Configuration(
                segmentation_strategy=SegmentationStrategy.PAGE,
                segment_processing=SegmentProcessing(
                    page=GenerationConfig(html=GenerationStrategy.LLM)
                ),
            ),
        )
        if client_type == "async"
        else client.upload(
            sample_path,
            Configuration(
                segmentation_strategy=SegmentationStrategy.PAGE,
                segment_processing=SegmentProcessing(
                    page=GenerationConfig(html=GenerationStrategy.LLM)
                ),
            ),
        )
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None


@pytest.mark.asyncio
async def test_page_llm(chunkr_client, sample_path):
    client_type, client = chunkr_client
    configuration = Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
        segment_processing=SegmentProcessing(
            page=GenerationConfig(
                html=GenerationStrategy.LLM, markdown=GenerationStrategy.LLM
            )
        ),
    )

    response = (
        await client.upload(sample_path, configuration)
        if client_type == "async"
        else client.upload(sample_path, configuration)
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None


@pytest.mark.asyncio
async def test_json_schema(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.upload(
            sample_path,
            Configuration(
                json_schema=JsonSchema(
                    title="Sales Data",
                    properties=[
                        Property(
                            name="Person with highest sales",
                            prop_type="string",
                            description="The person with the highest sales",
                        ),
                        Property(
                            name="Person with lowest sales",
                            prop_type="string",
                            description="The person with the lowest sales",
                        ),
                    ],
                )
            ),
        )
        if client_type == "async"
        else client.upload(
            sample_path,
            Configuration(
                json_schema=JsonSchema(
                    title="Sales Data",
                    properties=[
                        Property(
                            name="Person with highest sales",
                            prop_type="string",
                            description="The person with the highest sales",
                        ),
                        Property(
                            name="Person with lowest sales",
                            prop_type="string",
                            description="The person with the lowest sales",
                        ),
                    ],
                )
            ),
        )
    )

    assert response.task_id is not None
    if response.status != "Succeeded":
        raise ValueError(f"Task failed with message: {response.message}")
    assert response.output is not None


@pytest.mark.asyncio
async def test_delete_task(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.upload(sample_path)
        if client_type == "async"
        else client.upload(sample_path)
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

    if client_type == "async":
        await client.delete_task(response.task_id)
        with pytest.raises(Exception):
            await client.get_task(response.task_id)
    else:
        client.delete_task(response.task_id)
        with pytest.raises(Exception):
            client.get_task(response.task_id)


@pytest.mark.asyncio
async def test_delete_task_direct(chunkr_client, sample_path):
    client_type, client = chunkr_client
    task = (
        await client.upload(sample_path)
        if client_type == "async"
        else client.upload(sample_path)
    )
    assert isinstance(task, TaskResponse)
    assert task.task_id is not None
    assert task.status == "Succeeded"
    assert task.output is not None

    if client_type == "async":
        await client.delete_task(task.task_id)
        with pytest.raises(Exception):
            await client.get_task(task.task_id)
    else:
        client.delete_task(task.task_id)
        with pytest.raises(Exception):
            client.get_task(task.task_id)


@pytest.mark.asyncio
async def test_cancel_task(chunkr_client, sample_path):
    client_type, client = chunkr_client
    response = (
        await client.create_task(sample_path)
        if client_type == "async"
        else client.create_task(sample_path)
    )

    assert response.task_id is not None
    assert response.status == "Starting"

    if client_type == "async":
        await client.cancel_task(response.task_id)
        assert (await client.get_task(response.task_id)).status == "Cancelled"
        await response.poll()
    else:
        client.cancel_task(response.task_id)
        assert client.get_task(response.task_id).status == "Cancelled"
        response.poll()

    assert response.output is None


@pytest.mark.asyncio
async def test_cancel_task_direct(chunkr_client, sample_path):
    client_type, client = chunkr_client
    task = (
        await client.create_task(sample_path)
        if client_type == "async"
        else client.create_task(sample_path)
    )
    assert isinstance(task, TaskResponse)
    assert task.task_id is not None
    assert task.status == "Starting"

    if client_type == "async":
        await task.cancel_async()
    else:
        task.cancel()

    assert task.status == "Cancelled"
    assert task.output is None


@pytest.mark.asyncio
async def test_update_task(chunkr_client, sample_path):
    client_type, client = chunkr_client
    original_config = Configuration(
        segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
    )
    new_config = Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
    )
    response = (
        await client.upload(sample_path, original_config)
        if client_type == "async"
        else client.upload(sample_path, original_config)
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

    if client_type == "async":
        task = await client.update(response.task_id, new_config)
    else:
        task = client.update(response.task_id, new_config)

    assert task.status == "Succeeded"
    assert task.output is not None
    assert task.configuration.segmentation_strategy == SegmentationStrategy.PAGE


@pytest.mark.asyncio
async def test_update_task_direct(chunkr_client, sample_path):
    client_type, client = chunkr_client
    original_config = Configuration(
        segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
    )
    new_config = Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
    )
    task = (
        await client.upload(sample_path, original_config)
        if client_type == "async"
        else client.upload(sample_path, original_config)
    )
    assert isinstance(task, TaskResponse)
    assert task.task_id is not None
    assert task.status == "Succeeded"
    assert task.output is not None

    if client_type == "async":
        await task.update_async(new_config)
    else:
        task.update(new_config)

    assert task.status == "Succeeded"
    assert task.output is not None
    assert task.configuration.segmentation_strategy == SegmentationStrategy.PAGE


@pytest.mark.asyncio
async def test_pipeline_type(chunkr_client, sample_path, pipeline_type):
    client_type, client = chunkr_client
    response = (
        await client.upload(sample_path, Configuration(pipeline=pipeline_type))
        if client_type == "async"
        else client.upload(sample_path, Configuration(pipeline=pipeline_type))
    )

    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

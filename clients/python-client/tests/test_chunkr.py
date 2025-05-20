import pytest
from pathlib import Path
from PIL import Image
import asyncio
from typing import Awaitable

from chunkr_ai import Chunkr
from chunkr_ai.models import (
    ChunkProcessing,
    Configuration,
    EmbedSource,
    ErrorHandlingStrategy,
    FallbackStrategy,
    GenerationConfig,
    GenerationStrategy,
    LlmProcessing,
    OcrStrategy,
    Pipeline,
    SegmentationStrategy,
    SegmentProcessing,
    Status,
    TaskResponse,
    Tokenizer,
)

@pytest.fixture
def sample_path():
    return Path("tests/files/test.pdf")

@pytest.fixture
def sample_absolute_path_str():
    return "tests/files/test.pdf"

@pytest.fixture
def sample_relative_path_str():
    return "./tests/files/test.pdf"

@pytest.fixture
def sample_image():
    return Image.open("tests/files/test.jpg")

@pytest.fixture
def sample_url():
    return "https://chunkr-web.s3.us-east-1.amazonaws.com/landing_page/input/science.pdf"

@pytest.fixture
def client():
    client = Chunkr()
    yield client

@pytest.fixture(params=[
    pytest.param(None, id="none_pipeline"),
    pytest.param(Pipeline.AZURE, id="azure_pipeline"),
])
def pipeline_type(request):
    return request.param

@pytest.fixture
def markdown_embed_config():
    return Configuration(
        segment_processing=SegmentProcessing(
            Page=GenerationConfig(
                html=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM,
                embed_sources=[EmbedSource.MARKDOWN]
            )
        ),
    )

@pytest.fixture
def html_embed_config():
    return Configuration(
        segment_processing=SegmentProcessing(
            Page=GenerationConfig(
                html=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM,
                embed_sources=[EmbedSource.HTML]
            )
        ),
    )

@pytest.fixture
def multiple_embed_config():
    return Configuration(
        segment_processing=SegmentProcessing(
            Page=GenerationConfig(
                html=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM,
                llm="Generate a summary of this content",
                embed_sources=[EmbedSource.MARKDOWN, EmbedSource.LLM, EmbedSource.HTML]
            )
        ),
    )
    
@pytest.fixture
def word_tokenizer_string_config():
    return Configuration(
        chunk_processing=ChunkProcessing(
            tokenizer="Word"
        ),
    )

@pytest.fixture
def word_tokenizer_config():
    return Configuration(
        chunk_processing=ChunkProcessing(
            tokenizer=Tokenizer.WORD
        ),
    )

@pytest.fixture
def cl100k_tokenizer_config():
    return Configuration(
        chunk_processing=ChunkProcessing(
            tokenizer=Tokenizer.CL100K_BASE
        ),
    )

@pytest.fixture
def custom_tokenizer_config():
    return Configuration(
        chunk_processing=ChunkProcessing(
            tokenizer="Qwen/Qwen-tokenizer"
        ),
    )

@pytest.fixture
def xlm_roberta_with_html_content_config():
    return Configuration(
        chunk_processing=ChunkProcessing(
            tokenizer=Tokenizer.XLM_ROBERTA_BASE
        ),
        segment_processing=SegmentProcessing(
            Page=GenerationConfig(
                html=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM,
                embed_sources=[EmbedSource.HTML, EmbedSource.CONTENT]
            )
        ),
    )

@pytest.fixture
def none_fallback_config():
    return Configuration(
        llm_processing=LlmProcessing(
            model_id="gemini-pro-2.5",
            fallback_strategy=FallbackStrategy.none(),
            max_completion_tokens=500,
            temperature=0.2
        ),
    )

@pytest.fixture
def default_fallback_config():
    return Configuration(
        llm_processing=LlmProcessing(
            model_id="gemini-pro-2.5",
            fallback_strategy=FallbackStrategy.default(),
            max_completion_tokens=1000,
            temperature=0.5
        ),
    )

@pytest.fixture
def model_fallback_config():
    return Configuration(
        llm_processing=LlmProcessing(
            model_id="gemini-pro-2.5",
            fallback_strategy=FallbackStrategy.model("claude-3.7-sonnet"),
            max_completion_tokens=2000,
            temperature=0.7
        ),
    )

@pytest.fixture
def extended_context_config():
    return Configuration(
        segment_processing=SegmentProcessing(
            picture=GenerationConfig(
                extended_context=True,
                html=GenerationStrategy.LLM,
            ),
            table=GenerationConfig(
                extended_context=True,
                html=GenerationStrategy.LLM,
            )
        ),
    )

@pytest.mark.asyncio
async def test_ocr_auto(client, sample_path):
    response = await client.upload(sample_path, Configuration(ocr_strategy=OcrStrategy.AUTO))
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_expires_in(client, sample_path):
    response = await client.upload(sample_path, Configuration(expires_in=10))
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    await asyncio.sleep(11)
    try:
        await client.get_task(response.task_id)
        assert False
    except Exception as e:
        print(e)
        assert True

@pytest.mark.asyncio
async def test_chunk_processing(client, sample_path):
    response = await client.upload(
        sample_path,
        Configuration(chunk_processing=ChunkProcessing(target_length=1024)),
    )
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_segmentation_strategy_page(client, sample_path):
    response = await client.upload(
        sample_path, Configuration(segmentation_strategy=SegmentationStrategy.PAGE)
    )
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_page_llm_html(client, sample_path):
    response = await client.upload(
        sample_path,
        Configuration(
            segmentation_strategy=SegmentationStrategy.PAGE,
            segment_processing=SegmentProcessing(
                Page=GenerationConfig(html=GenerationStrategy.LLM)
            ),
        ),
    )
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_page_llm(client, sample_path):
    configuration = Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
        segment_processing=SegmentProcessing(
            Page=GenerationConfig(
                html=GenerationStrategy.LLM, markdown=GenerationStrategy.LLM
            )
        ),
    )
    response = await client.upload(sample_path, configuration)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_delete_task(client, sample_path):
    response = await client.upload(sample_path)
    await client.delete_task(response.task_id)
    with pytest.raises(Exception):
        await client.get_task(response.task_id)

@pytest.mark.asyncio
async def test_delete_task_direct(client, sample_path):
    task = await client.upload(sample_path)
    assert isinstance(task, TaskResponse)
    assert task.task_id is not None
    assert task.status == "Succeeded"
    assert task.output is not None
    await client.delete_task(task.task_id)
    with pytest.raises(Exception):
        await client.get_task(task.task_id)

@pytest.mark.asyncio
async def test_cancel_task(client, sample_path):
    response = await client.create_task(sample_path)
    assert response.status == "Starting"
    await client.cancel_task(response.task_id)
    assert (await client.get_task(response.task_id)).status == "Cancelled"

@pytest.mark.asyncio
async def test_cancel_task_direct(client, sample_path):
    task = await client.create_task(sample_path)
    assert task.status == "Starting"
    try:
        await task.cancel()
    except Exception as e:
        task = await client.get_task(task.task_id)
        print(task.status)
        if task.status == Status.PROCESSING:
            print("Task is processing, so it can't be cancelled")
            assert True
        else:
            print("Task status:", task.status)
            raise e
    assert task.status == "Cancelled"

@pytest.mark.asyncio
async def test_update_task(client, sample_path):
    original_config = Configuration(
        segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
    )
    new_config = Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
    )
    task = await client.upload(sample_path, original_config)
    task = await client.update(task.task_id, new_config)
    assert task.status == "Succeeded"
    assert task.output is not None
    assert task.configuration.segmentation_strategy == SegmentationStrategy.PAGE

@pytest.mark.asyncio
async def test_update_task_direct(client, sample_path):
    original_config = Configuration(
        segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
    )
    new_config = Configuration(
        segmentation_strategy=SegmentationStrategy.PAGE,
    )
    task = await client.upload(sample_path, original_config)
    task = await (await task.update(new_config))
    assert isinstance(task, TaskResponse)
    assert task.status == "Succeeded"
    assert task.output is not None
    assert task.configuration.segmentation_strategy == SegmentationStrategy.PAGE

@pytest.mark.asyncio
async def test_pipeline_type_azure(client, sample_path):
    response = await client.upload(sample_path, Configuration(pipeline=Pipeline.AZURE))
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    assert response.configuration.pipeline == Pipeline.AZURE
    
@pytest.mark.asyncio
async def test_pipeline_type_chunkr(client, sample_path):
    response = await client.upload(sample_path, Configuration(pipeline=Pipeline.CHUNKR))
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    assert response.configuration.pipeline == Pipeline.CHUNKR
    
@pytest.mark.asyncio
async def test_client_lifecycle(client, sample_path):
    response1 = await client.upload(sample_path)
    await client.close()
    response2 = await client.upload(sample_path)
    assert response1.task_id is not None
    assert response2.task_id is not None

@pytest.mark.asyncio
async def test_task_operations_after_client_close(client, sample_path):
    task = await client.create_task(sample_path)
    await client.close()
    result = await task.poll()
    assert result.status == "Succeeded"

@pytest.mark.asyncio
async def test_output_files_no_dir(client, sample_path, tmp_path):
    task = await client.upload(sample_path)
    
    html_file = tmp_path / "output.html"
    md_file = tmp_path / "output.md"
    content_file = tmp_path / "output.txt"
    json_file = tmp_path / "output.json"
    
    task.html(html_file)
    task.markdown(md_file)
    task.content(content_file)
    task.json(json_file)
    
    assert html_file.exists()
    assert md_file.exists()
    assert content_file.exists()
    assert json_file.exists()

@pytest.mark.asyncio
async def test_output_files_with_dirs(client, sample_path, tmp_path):
    task = await client.upload(sample_path)
    
    nested_dir = tmp_path / "nested" / "output" / "dir"
    html_file = nested_dir / "output.html"
    md_file = nested_dir / "output.md"
    content_file = nested_dir / "output.txt"
    json_file = nested_dir / "output.json"
    
    task.html(html_file)
    task.markdown(md_file)
    task.content(content_file)
    task.json(json_file)

    assert html_file.exists()
    assert md_file.exists()
    assert content_file.exists()
    assert json_file.exists()


@pytest.mark.asyncio
async def test_combined_config_with_llm_and_other_settings(client, sample_path):
    # Test combining LLM settings with other configuration options
    config = Configuration(
        llm_processing=LlmProcessing(
            model_id="qwen-2.5-vl-7b-instruct",
            fallback_strategy=FallbackStrategy.model("gemini-flash-2.0"),
            temperature=0.4
        ),
        segmentation_strategy=SegmentationStrategy.PAGE,
        segment_processing=SegmentProcessing(
            Page=GenerationConfig(
                html=GenerationStrategy.LLM,
                markdown=GenerationStrategy.LLM
            )
        ),
        chunk_processing=ChunkProcessing(target_length=1024)
    )
    
    response = await client.upload(sample_path, config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    assert response.configuration.llm_processing is not None
    assert response.configuration.llm_processing.model_id == "qwen-2.5-vl-7b-instruct"
    assert response.configuration.segmentation_strategy == SegmentationStrategy.PAGE
    assert response.configuration.chunk_processing.target_length == 1024

@pytest.mark.asyncio
async def test_embed_sources_markdown_only(client, sample_path, markdown_embed_config):
    response = await client.upload(sample_path, markdown_embed_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    # Check the first chunk to verify embed exists
    if response.output.chunks:
        chunk = response.output.chunks[0]
        assert chunk.embed is not None

@pytest.mark.asyncio
async def test_embed_sources_html_only(client, sample_path, html_embed_config):
    response = await client.upload(sample_path, html_embed_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_embed_sources_multiple(client, sample_path, multiple_embed_config):
    response = await client.upload(sample_path, multiple_embed_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_tokenizer_word(client, sample_path, word_tokenizer_config):
    response = await client.upload(sample_path, word_tokenizer_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    if response.output.chunks:
        for chunk in response.output.chunks:
            # Word tokenizer should result in chunks with length close to target
            assert chunk.chunk_length > 0
            assert chunk.chunk_length <= 600  # Allow some flexibility

@pytest.mark.asyncio
async def test_tokenizer_cl100k(client, sample_path, cl100k_tokenizer_config):
    response = await client.upload(sample_path, cl100k_tokenizer_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_tokenizer_custom_string(client, sample_path, custom_tokenizer_config):
    response = await client.upload(sample_path, custom_tokenizer_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_embed_sources_with_different_tokenizer(client, sample_path, xlm_roberta_with_html_content_config):
    response = await client.upload(sample_path, xlm_roberta_with_html_content_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_error_handling_continue(client, sample_path):
    response = await client.upload(sample_path, Configuration(error_handling=ErrorHandlingStrategy.CONTINUE))
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None

@pytest.mark.asyncio
async def test_llm_processing_none_fallback(client, sample_path, none_fallback_config):
    response = await client.upload(sample_path, none_fallback_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    assert response.configuration.llm_processing is not None
    assert response.configuration.llm_processing.model_id == "gemini-pro-2.5"
    assert str(response.configuration.llm_processing.fallback_strategy) == "None"
    assert response.configuration.llm_processing.max_completion_tokens == 500
    assert response.configuration.llm_processing.temperature == 0.2

@pytest.mark.asyncio
async def test_llm_processing_default_fallback(client, sample_path, default_fallback_config):
    response = await client.upload(sample_path, default_fallback_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    assert response.configuration.llm_processing is not None
    assert response.configuration.llm_processing.model_id == "gemini-pro-2.5"
    # The service may resolve Default to an actual model
    assert response.configuration.llm_processing.fallback_strategy is not None
    assert response.configuration.llm_processing.max_completion_tokens == 1000
    assert response.configuration.llm_processing.temperature == 0.5

@pytest.mark.asyncio
async def test_llm_processing_model_fallback(client, sample_path, model_fallback_config):
    response = await client.upload(sample_path, model_fallback_config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    assert response.configuration.llm_processing is not None
    assert response.configuration.llm_processing.model_id == "gemini-pro-2.5"
    assert str(response.configuration.llm_processing.fallback_strategy) == "Model(claude-3.7-sonnet)"
    assert response.configuration.llm_processing.max_completion_tokens == 2000
    assert response.configuration.llm_processing.temperature == 0.7

@pytest.mark.asyncio
async def test_llm_custom_model(client, sample_path):
    config = Configuration(
        llm_processing=LlmProcessing(
            model_id="claude-3.7-sonnet",  # Using a model from models.yaml
            fallback_strategy=FallbackStrategy.none(),
            max_completion_tokens=1500,
            temperature=0.3
        ),
    )
    response = await client.upload(sample_path, config)
    assert response.task_id is not None
    assert response.status == "Succeeded"
    assert response.output is not None
    assert response.configuration.llm_processing is not None
    assert response.configuration.llm_processing.model_id == "claude-3.7-sonnet"

@pytest.mark.asyncio
async def test_fallback_strategy_serialization():
    # Test that FallbackStrategy objects serialize correctly
    none_strategy = FallbackStrategy.none()
    default_strategy = FallbackStrategy.default()
    model_strategy = FallbackStrategy.model("gpt-4.1")
    
    assert none_strategy.model_dump() == "None"
    assert default_strategy.model_dump() == "Default"
    assert model_strategy.model_dump() == {"Model": "gpt-4.1"}
    
    # Test string representation
    assert str(none_strategy) == "None"
    assert str(default_strategy) == "Default"
    assert str(model_strategy) == "Model(gpt-4.1)"

@pytest.mark.asyncio
async def test_extended_context(client, sample_path, extended_context_config):
    """Tests uploading with extended context enabled for pictures and tables."""
    print("\nTesting extended context for Pictures and Tables...")
    try:
        task = await client.upload(sample_path, config=extended_context_config)
        print(f"Task created with extended context config: {task.task_id}")
        print(f"Initial Status: {task.status}")

        # Poll the task until it finishes or fails
        print(f"Final Status: {task.status}")
        print(f"Message: {task.message}")

        # Basic assertion: Check if the task completed (either succeeded or failed)
        assert task.status in [Status.SUCCEEDED, Status.FAILED], f"Task ended in unexpected state: {task.status}"

        # More specific assertions based on expected outcomes with your local server
        # if task.status == Status.FAILED:
        #     assert "context_length_exceeded" in task.message, "Expected context length error"
        # elif task.status == Status.SUCCEEDED:
        #     # Check if output reflects extended context usage if possible
        #     pass

        print("Extended context test completed.")

    except Exception as e:
        print(f"Error during extended context test: {e}")
        raise # Re-raise the exception to fail the test explicitly
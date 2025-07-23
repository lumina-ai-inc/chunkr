import pytest
from pathlib import Path

from chunkr_ai import Chunkr
from chunkr_ai.models import (
    Configuration,
    ChunkProcessing,
    SegmentProcessing,
    GenerationConfig,
    SegmentFormat,
    GenerationStrategy,
    EmbedSource,
    Tokenizer,
    OcrStrategy,
    SegmentationStrategy,
    Page,
)


@pytest.fixture
def client():
    """Chunkr client instance"""
    client = Chunkr()
    yield client


@pytest.fixture
def sample_pdf_path():
    """Path to the PDF test file"""
    return Path("tests/files/test.pdf")


@pytest.fixture
def excel_sample_path():
    """Path to the Excel test file"""
    return Path("tests/files/excel/test.xlsx")


@pytest.fixture
def basic_config():
    """Basic configuration for testing pages"""
    return Configuration(
        high_resolution=True,
        ocr_strategy=OcrStrategy.ALL,
        segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
    )


class TestPagesBasicFunctionality:
    """Test basic pages functionality across different file types"""

    @pytest.mark.asyncio
    async def test_pdf_generates_pages(self, client, sample_pdf_path, basic_config):
        """Test that PDF files generate pages information"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        assert response.task_id is not None
        assert response.status == "Succeeded"
        assert response.output is not None
        
        # Test pages structure
        if response.output.pages:  # Pages might be optional for some file types
            assert len(response.output.pages) > 0
            page = response.output.pages[0]
            assert page.image is not None
            assert page.page_number is not None
            assert page.page_height is not None
            assert page.page_width is not None

    @pytest.mark.asyncio
    async def test_excel_generates_pages_with_sheet_info(self, client, excel_sample_path, basic_config):
        """Test that Excel files generate pages with sheet information"""
        response = await client.upload(excel_sample_path, basic_config)
        
        assert response.task_id is not None
        assert response.status == "Succeeded"
        assert response.output is not None
        
        # Excel should definitely have pages
        assert response.output.pages is not None
        assert len(response.output.pages) > 0
        
        page = response.output.pages[0]
        assert page.image is not None
        assert page.page_number is not None
        assert page.page_height is not None
        assert page.page_width is not None
        # Excel pages should have sheet names
        assert page.ss_sheet_name is not None

    @pytest.mark.asyncio
    async def test_page_count_consistency(self, client, sample_pdf_path, basic_config):
        """Test that page_count matches the actual number of pages"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        assert response.output.page_count is not None
        if response.output.pages:
            assert response.output.page_count == len(response.output.pages)
        else:
            # If no pages array, page_count should still be meaningful
            assert response.output.page_count > 0


class TestPageStructure:
    """Test the Page model structure and validation"""

    @pytest.mark.asyncio
    async def test_page_required_fields(self, client, sample_pdf_path, basic_config):
        """Test that Page objects have all required fields"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        if response.output.pages and len(response.output.pages) > 0:
            page = response.output.pages[0]
            
            # Required fields
            assert page.image is not None
            assert isinstance(page.page_number, int)
            assert isinstance(page.page_height, (int, float))
            assert isinstance(page.page_width, (int, float))
            
            # Optional fields should be accessible
            assert hasattr(page, 'ss_sheet_name')

    @pytest.mark.asyncio
    async def test_page_numbers_start_from_one(self, client, sample_pdf_path, basic_config):
        """Test that page numbers start from 1 and are sequential"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        if response.output.pages and len(response.output.pages) > 0:
            page_numbers = [page.page_number for page in response.output.pages]
            page_numbers.sort()
            
            # Should start from 1
            assert page_numbers[0] == 1, f"Page numbers should start from 1, got: {page_numbers[0]}"
            
            # Should be sequential if multiple pages
            if len(page_numbers) > 1:
                for i in range(1, len(page_numbers)):
                    assert page_numbers[i] == page_numbers[i-1] + 1, f"Page numbers not sequential: {page_numbers}"

    @pytest.mark.asyncio
    async def test_page_dimensions_positive(self, client, sample_pdf_path, basic_config):
        """Test that page dimensions are positive values"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        if response.output.pages:
            for page in response.output.pages:
                assert page.page_height > 0, f"Invalid page height: {page.page_height}"
                assert page.page_width > 0, f"Invalid page width: {page.page_width}"

    @pytest.mark.asyncio
    async def test_page_images_are_urls(self, client, sample_pdf_path, basic_config):
        """Test that page images are valid URLs"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        if response.output.pages:
            for page in response.output.pages:
                assert page.image.startswith(('http://', 'https://')), f"Invalid page image URL: {page.image}"


class TestMimeTypeHandling:
    """Test MIME type handling for different file types"""

    @pytest.mark.asyncio
    async def test_pdf_mime_type(self, client, sample_pdf_path, basic_config):
        """Test that PDF files have correct MIME type"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        assert response.output.mime_type is not None
        # Should be PDF MIME type
        assert response.output.mime_type in [
            "application/pdf",
            "application/x-pdf"
        ], f"Unexpected PDF MIME type: {response.output.mime_type}"

    @pytest.mark.asyncio
    async def test_excel_mime_type(self, client, excel_sample_path, basic_config):
        """Test that Excel files have correct MIME type"""
        response = await client.upload(excel_sample_path, basic_config)
        
        assert response.output.mime_type is not None
        # Should be Excel MIME type
        expected_excel_types = [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel"
        ]
        assert response.output.mime_type in expected_excel_types, f"Unexpected Excel MIME type: {response.output.mime_type}"


class TestBackwardsCompatibility:
    """Test that new fields don't break existing functionality"""

    @pytest.mark.asyncio
    async def test_existing_fields_still_work(self, client, sample_pdf_path, basic_config):
        """Test that all existing fields still work with new page functionality"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        # Test that all traditional fields still work
        assert response.task_id is not None
        assert response.status == "Succeeded"
        assert response.output is not None
        assert response.output.chunks is not None
        assert len(response.output.chunks) > 0
        assert response.output.file_name is not None
        assert response.output.page_count is not None
        assert response.output.pdf_url is not None
        
        # Test chunk structure
        chunk = response.output.chunks[0]
        assert chunk.chunk_id is not None
        assert chunk.chunk_length is not None
        assert chunk.segments is not None
        assert len(chunk.segments) > 0
        
        # Test segment structure
        segment = chunk.segments[0]
        assert segment.segment_id is not None
        assert segment.segment_type is not None
        assert segment.bbox is not None

    @pytest.mark.asyncio
    async def test_optional_new_fields(self, client, sample_pdf_path, basic_config):
        """Test that new optional fields are properly handled"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        # New fields should be accessible but might be None
        assert hasattr(response.output, 'mime_type')
        assert hasattr(response.output, 'pages')
        
        # For segments, spreadsheet fields should be accessible but None for PDFs
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                assert hasattr(segment, 'ss_cells')
                assert hasattr(segment, 'ss_range')
                assert hasattr(segment, 'ss_sheet_name')
                assert hasattr(segment, 'segment_length')
                
                # For non-Excel files, these should be None
                if response.output.mime_type != "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    assert segment.ss_cells is None
                    assert segment.ss_range is None
                    assert segment.ss_sheet_name is None


class TestErrorHandling:
    """Test error handling for pages functionality"""

    @pytest.mark.asyncio
    async def test_missing_pages_handled_gracefully(self, client, sample_pdf_path, basic_config):
        """Test that missing pages are handled gracefully"""
        response = await client.upload(sample_pdf_path, basic_config)
        
        # Even if pages is None, the response should be valid
        if response.output.pages is None:
            # page_count should still be available
            assert response.output.page_count is not None
            assert response.output.page_count > 0
        else:
            # If pages exist, they should be valid
            assert len(response.output.pages) > 0
            assert response.output.page_count == len(response.output.pages) 
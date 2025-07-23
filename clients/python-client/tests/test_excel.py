import pytest
import json
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
    Cell,
    CellStyle,
    Alignment,
    VerticalAlignment,
    Page,
    Segment,
    SegmentType,
)


@pytest.fixture
def excel_sample_path():
    """Path to the Excel test file"""
    return Path("tests/files/excel/test.xlsx")


@pytest.fixture
def excel_expected_output():
    """Expected output for Excel test file"""
    with open("tests/files/excel/test.json", "r") as f:
        return json.load(f)


@pytest.fixture
def client():
    """Chunkr client instance"""
    client = Chunkr()
    yield client


@pytest.fixture
def excel_config():
    """Configuration optimized for Excel processing"""
    return Configuration(
        high_resolution=True,
        ocr_strategy=OcrStrategy.ALL,
        segmentation_strategy=SegmentationStrategy.LAYOUT_ANALYSIS,
        chunk_processing=ChunkProcessing(
            target_length=512,
            tokenizer=Tokenizer.WORD,
        ),
        segment_processing=SegmentProcessing(
            Table=GenerationConfig(
                format=SegmentFormat.MARKDOWN,
                strategy=GenerationStrategy.AUTO,
                embed_sources=[EmbedSource.MARKDOWN],
            ),
            Text=GenerationConfig(
                format=SegmentFormat.MARKDOWN,
                strategy=GenerationStrategy.AUTO,
                embed_sources=[EmbedSource.MARKDOWN],
            ),
        ),
    )


class TestExcelBasicFunctionality:
    """Test basic Excel file processing"""

    @pytest.mark.asyncio
    async def test_excel_upload_and_process(self, client, excel_sample_path, excel_config):
        """Test that Excel file can be uploaded and processed successfully"""
        response = await client.upload(excel_sample_path, excel_config)
        
        assert response.task_id is not None
        assert response.status == "Succeeded"
        assert response.output is not None
        assert response.output.chunks is not None
        assert len(response.output.chunks) > 0

    @pytest.mark.asyncio
    async def test_excel_mime_type(self, client, excel_sample_path, excel_config):
        """Test that Excel files have correct MIME type"""
        response = await client.upload(excel_sample_path, excel_config)
        
        assert response.output.mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    @pytest.mark.asyncio
    async def test_excel_pages_exist(self, client, excel_sample_path, excel_config):
        """Test that Excel processing generates pages information"""
        response = await client.upload(excel_sample_path, excel_config)
        
        assert response.output.pages is not None
        assert len(response.output.pages) > 0
        assert response.output.page_count is not None
        assert response.output.page_count > 0

    @pytest.mark.asyncio
    async def test_excel_chunks_have_segments(self, client, excel_sample_path, excel_config):
        """Test that Excel chunks contain segments with data"""
        response = await client.upload(excel_sample_path, excel_config)
        
        assert len(response.output.chunks) > 0
        chunk = response.output.chunks[0]
        assert len(chunk.segments) > 0
        assert chunk.chunk_length > 0


class TestExcelSpreadsheetFields:
    """Test Excel-specific spreadsheet fields"""

    @pytest.mark.asyncio
    async def test_segments_have_spreadsheet_fields(self, client, excel_sample_path, excel_config):
        """Test that segments contain spreadsheet-specific fields"""
        response = await client.upload(excel_sample_path, excel_config)
        
        # Find a segment with spreadsheet data
        spreadsheet_segment = None
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                if segment.ss_cells and len(segment.ss_cells) > 0:
                    spreadsheet_segment = segment
                    break
            if spreadsheet_segment:
                break
        
        assert spreadsheet_segment is not None, "No segment with spreadsheet data found"
        
        # Test spreadsheet-specific fields
        assert spreadsheet_segment.ss_cells is not None
        assert len(spreadsheet_segment.ss_cells) > 0
        assert spreadsheet_segment.ss_sheet_name is not None
        assert spreadsheet_segment.ss_range is not None

    @pytest.mark.asyncio
    async def test_cells_have_required_fields(self, client, excel_sample_path, excel_config):
        """Test that cells contain all required fields"""
        response = await client.upload(excel_sample_path, excel_config)
        
        # Find a segment with cells
        test_cell = None
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                if segment.ss_cells and len(segment.ss_cells) > 0:
                    test_cell = segment.ss_cells[0]
                    break
            if test_cell:
                break
        
        assert test_cell is not None, "No cell found in any segment"
        
        # Test required cell fields
        assert test_cell.cell_id is not None
        assert test_cell.text is not None
        assert test_cell.range is not None
        # Optional fields should be accessible
        assert hasattr(test_cell, 'formula')
        assert hasattr(test_cell, 'value')
        assert hasattr(test_cell, 'hyperlink')
        assert hasattr(test_cell, 'style')

    @pytest.mark.asyncio
    async def test_cell_styling_fields(self, client, excel_sample_path, excel_config):
        """Test that cells with styling contain CellStyle information"""
        response = await client.upload(excel_sample_path, excel_config)
        
        # Find a cell with styling
        styled_cell = None
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                if segment.ss_cells:
                    for cell in segment.ss_cells:
                        if cell.style is not None:
                            styled_cell = cell
                            break
                if styled_cell:
                    break
            if styled_cell:
                break
        
        assert styled_cell is not None, "No styled cell found"
        assert styled_cell.style is not None
        
        # Test CellStyle fields
        style = styled_cell.style
        assert hasattr(style, 'bg_color')
        assert hasattr(style, 'text_color')
        assert hasattr(style, 'font_face')
        assert hasattr(style, 'is_bold')
        assert hasattr(style, 'align')
        assert hasattr(style, 'valign')

    @pytest.mark.asyncio
    async def test_excel_sheet_names(self, client, excel_sample_path, excel_config):
        """Test that sheet names are properly captured"""
        response = await client.upload(excel_sample_path, excel_config)
        
        # Check segments for sheet names
        sheet_names = set()
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                if segment.ss_sheet_name:
                    sheet_names.add(segment.ss_sheet_name)
        
        assert len(sheet_names) > 0, "No sheet names found in segments"
        
        # Check pages for sheet names
        page_sheet_names = set()
        if response.output.pages:
            for page in response.output.pages:
                if page.ss_sheet_name:
                    page_sheet_names.add(page.ss_sheet_name)
        
        # At least one source should have sheet names
        assert len(sheet_names) > 0 or len(page_sheet_names) > 0


class TestExcelPages:
    """Test Excel pages functionality"""

    @pytest.mark.asyncio
    async def test_pages_structure(self, client, excel_sample_path, excel_config):
        """Test that pages have correct structure and fields"""
        response = await client.upload(excel_sample_path, excel_config)
        
        assert response.output.pages is not None
        assert len(response.output.pages) > 0
        
        page = response.output.pages[0]
        assert page.image is not None
        assert page.page_number is not None
        assert page.page_height is not None
        assert page.page_width is not None
        # ss_sheet_name is optional for pages
        assert hasattr(page, 'ss_sheet_name')

    @pytest.mark.asyncio
    async def test_page_count_consistency(self, client, excel_sample_path, excel_config):
        """Test that page_count matches the actual number of pages"""
        response = await client.upload(excel_sample_path, excel_config)
        
        assert response.output.page_count is not None
        if response.output.pages:
            assert response.output.page_count == len(response.output.pages)

    @pytest.mark.asyncio
    async def test_page_numbers_sequential(self, client, excel_sample_path, excel_config):
        """Test that page numbers are sequential and start from 1"""
        response = await client.upload(excel_sample_path, excel_config)
        
        if response.output.pages and len(response.output.pages) > 1:
            page_numbers = [page.page_number for page in response.output.pages]
            page_numbers.sort()
            
            # Should start from 1 and be sequential
            for i, page_num in enumerate(page_numbers):
                assert page_num == i + 1, f"Page numbers not sequential: {page_numbers}"


class TestExcelSegmentTypes:
    """Test Excel segment types and their properties"""

    @pytest.mark.asyncio
    async def test_segment_types_present(self, client, excel_sample_path, excel_config):
        """Test that appropriate segment types are detected in Excel files"""
        response = await client.upload(excel_sample_path, excel_config)
        
        segment_types = set()
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                segment_types.add(segment.segment_type)
        
        # Excel files should contain at least Table or Text segments
        expected_types = {SegmentType.TABLE, SegmentType.TEXT}
        assert len(segment_types.intersection(expected_types)) > 0, f"No expected segment types found. Got: {segment_types}"

    @pytest.mark.asyncio
    async def test_table_segments_have_cells(self, client, excel_sample_path, excel_config):
        """Test that TABLE segments contain cell data"""
        response = await client.upload(excel_sample_path, excel_config)
        
        table_segments = []
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                if segment.segment_type == SegmentType.TABLE:
                    table_segments.append(segment)
        
        if table_segments:  # If we have table segments, they should have cells
            found_cells = False
            for segment in table_segments:
                if segment.ss_cells and len(segment.ss_cells) > 0:
                    found_cells = True
                    break
            assert found_cells, "TABLE segments should contain cell data"


class TestExcelEmbedding:
    """Test Excel embedding functionality"""

    @pytest.mark.asyncio
    async def test_chunks_have_embed_content(self, client, excel_sample_path, excel_config):
        """Test that chunks generate embed content for Excel data"""
        response = await client.upload(excel_sample_path, excel_config)
        
        # At least some chunks should have embed content
        chunks_with_embed = [chunk for chunk in response.output.chunks if chunk.embed]
        assert len(chunks_with_embed) > 0, "No chunks with embed content found"
        
        # Embed content should not be empty
        for chunk in chunks_with_embed:
            assert len(chunk.embed.strip()) > 0, "Empty embed content found"

    @pytest.mark.asyncio
    async def test_segment_length_calculation(self, client, excel_sample_path, excel_config):
        """Test that segments have length calculations"""
        response = await client.upload(excel_sample_path, excel_config)
        
        segments_with_length = []
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                if segment.segment_length is not None:
                    segments_with_length.append(segment)
        
        # At least some segments should have length calculations
        assert len(segments_with_length) > 0, "No segments with length calculations found"
        
        # Lengths should be positive
        for segment in segments_with_length:
            assert segment.segment_length > 0, f"Invalid segment length: {segment.segment_length}"


class TestExcelEdgeCases:
    """Test edge cases and error handling for Excel processing"""

    @pytest.mark.asyncio
    async def test_empty_cells_handling(self, client, excel_sample_path, excel_config):
        """Test that empty cells are handled properly"""
        response = await client.upload(excel_sample_path, excel_config)
        
        # Look for cells that might be empty
        all_cells = []
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                if segment.ss_cells:
                    all_cells.extend(segment.ss_cells)
        
        assert len(all_cells) > 0, "No cells found to test"
        
        # All cells should have a text field, even if empty
        for cell in all_cells:
            assert hasattr(cell, 'text'), "Cell missing text field"
            assert cell.text is not None, "Cell text is None"

    @pytest.mark.asyncio
    async def test_range_format_validity(self, client, excel_sample_path, excel_config):
        """Test that Excel ranges follow expected format"""
        response = await client.upload(excel_sample_path, excel_config)
        
        ranges = []
        for chunk in response.output.chunks:
            for segment in chunk.segments:
                if segment.ss_range:
                    ranges.append(segment.ss_range)
                if segment.ss_cells:
                    for cell in segment.ss_cells:
                        ranges.append(cell.range)
        
        assert len(ranges) > 0, "No ranges found to test"
        
        # Basic range format validation (e.g., "A1", "A1:B2")
        import re
        range_pattern = re.compile(r'^[A-Z]+\d+(:[A-Z]+\d+)?$')
        valid_ranges = [r for r in ranges if range_pattern.match(r)]
        
        # Most ranges should follow the expected format
        assert len(valid_ranges) > 0, f"No valid ranges found. Ranges: {ranges[:10]}..."


# Integration test using the expected output fixture
class TestExcelIntegration:
    """Integration tests comparing against expected output"""

    @pytest.mark.asyncio
    async def test_compare_with_expected_structure(self, client, excel_sample_path, excel_config, excel_expected_output):
        """Test that the output structure matches expected format"""
        response = await client.upload(excel_sample_path, excel_config)
        
        expected = excel_expected_output["output"]
        actual = response.output
        
        # Compare high-level structure
        assert actual.mime_type == expected["mime_type"]
        assert actual.page_count == expected["page_count"]
        assert len(actual.chunks) > 0
        assert len(actual.pages) > 0
        
        # Verify that we have similar data structure
        expected_has_cells = any(
            segment.get("ss_cells") 
            for chunk in expected["chunks"] 
            for segment in chunk["segments"]
        )
        actual_has_cells = any(
            segment.ss_cells 
            for chunk in actual.chunks 
            for segment in chunk.segments 
            if segment.ss_cells
        )
        
        if expected_has_cells:
            assert actual_has_cells, "Expected cells in output but none found" 
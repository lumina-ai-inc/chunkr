# Python Client Tests

This directory contains test files for the Chunkr Python client.

## Test Files

- `test_chunkr.py` - Main test suite for core functionality and backwards compatibility
- `test_excel.py` - Comprehensive tests for Excel/spreadsheet functionality 
- `test_pages.py` - Tests for pages functionality across different file types
- `test_file_handling.py` - File handling and upload tests
- `bbox_visualizer.py` - Bounding box visualization utilities
- `main.py` - Test runner and examples

## Test Fixtures

- `files/test.pdf` - Sample PDF for testing
- `files/test.jpg` - Sample image for testing  
- `files/excel/test.xlsx` - Sample Excel file for spreadsheet testing
- `files/excel/test.json` - Expected output for Excel test file

## New Features Tested

### Excel/Spreadsheet Features
- Cell-level data extraction with formulas, values, and styling
- Sheet name handling
- Excel range processing
- Cell styling (colors, fonts, alignment)
- Spreadsheet-specific MIME types

### Pages Features  
- Page metadata and images
- Page count consistency
- Sheet name association with pages
- MIME type detection for different file formats

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_excel.py
pytest tests/test_pages.py
pytest tests/test_chunkr.py

# Run with verbose output
pytest tests/ -v

# Run specific test classes
pytest tests/test_excel.py::TestExcelBasicFunctionality
pytest tests/test_pages.py::TestPagesBasicFunctionality
```

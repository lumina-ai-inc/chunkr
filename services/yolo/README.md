# YOLO Document Layout Segmentation Service

This service provides document layout analysis capabilities using the DocLayout-YOLO model. It detects and classifies different elements in document images such as titles, text blocks, tables, figures, and formulas.

## Overview

The service is built on top of the [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) model, which is a real-time and robust layout detection model for diverse documents based on YOLO-v10. The implementation uses FastAPI to provide a REST API for document layout analysis.

## Features

- Document layout detection and segmentation
- Classification of document elements into categories:
  - Title
  - Text
  - Page Header/Footer
  - Picture/Figure
  - Caption
  - Table
  - Footnote
  - Formula
- Batch processing support
- Asynchronous API for handling large documents

## API Endpoints

### `GET /`

Health check endpoint that returns a simple message indicating the service is running.

### `POST /batch_async`

Process a single document image asynchronously.

**Parameters:**
- `file`: The document image file
- `ocr_data`: OCR data in JSON format (optional)

**Returns:**
- JSON object containing detected layout elements with bounding boxes, scores, and classes

### `POST /batch`

Process multiple document images in a batch.

**Parameters:**
- `files`: List of document image files
- `ocr_data`: OCR data in JSON format (optional)

**Returns:**
- JSON array containing detected layout elements for each image

## Class Mapping

The service maps YOLO classes to segment types used in the application:

| YOLO Class | Description | Mapped Segment Type |
|------------|-------------|---------------------|
| 0 | Title | Title (10) |
| 1 | Plain Text | Text (9) |
| 2 | Abandon | PageHeader (5) / PageFooter (4) / Text (9) |
| 3 | Figure | Picture (6) |
| 4 | Figure Caption | Caption (0) |
| 5 | Table | Table (8) |
| 6 | Table Caption | Caption (0) |
| 7 | Table Footnote | Footnote (1) |
| 8 | Isolate Formula | Formula (2) |
| 9 | Formula Caption | Caption (0) |


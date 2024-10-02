# # For APACHE 2.0:

# The following changes have been made to improve the API:
# 1. Routes have been refactored for convenience.
# 2. A new 'density' parameter has been added to the '/analyze/high-quality' route.
# 3. The '/analyze/fast' route now uses run_in_threadpool for better performance.
# 4. A processing lock has been implemented for the '/analyze/high-quality' route to prevent concurrent processing.

# New routes:
# - /readiness: GET request to check if the service is ready
# - /: GET request to get system information
# - /analyze/fast: POST request for fast PDF analysis
# - /analyze/high-quality: POST request for high-quality PDF analysis with density and extension parameters

# Removed routes:
# - /save_xml/{xml_file_name}: POST request to analyze and save XML
# - /get_xml/{xml_file_name}: GET request to retrieve XML by name
# - /toc: POST request to get table of contents
# - /text: POST request to get text extraction
# - /error: GET request to test error handling
# - /: POST request for general PDF analysis (replaced by specific /analyze routes)

import sys
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from asyncio import Lock
from fastapi.concurrency import run_in_threadpool

from catch_exceptions import catch_exceptions
from configuration import service_logger
from pdf_layout_analysis.get_xml import get_xml
from pdf_layout_analysis.run_pdf_layout_analysis import analyze_pdf
from pdf_layout_analysis.run_pdf_layout_analysis_fast import analyze_pdf_fast
from text_extraction.get_text_extraction import get_text_extraction
from toc.get_toc import get_toc

service_logger.info(f"Is PyTorch using GPU: {torch.cuda.is_available()}")

app = FastAPI()
processing_lock = Lock()


@app.get("/readiness")
def readiness():
    return {"status": "ready"}


@app.get("/")
async def info():
    return sys.version + " Using GPU: " + str(torch.cuda.is_available())

@app.post("/analyze/fast")
@catch_exceptions
async def run_fast(file: UploadFile = File(...)):
    return await run_in_threadpool(analyze_pdf_fast, file.file.read(), "")


@app.post("/analyze/high-quality")
@catch_exceptions
async def run_high_quality(
    file: UploadFile = File(...),
    density: int = Form(72),
    extension: str = Form("jpeg")
):
    async with processing_lock:
        return analyze_pdf(file.file.read(), "", density, extension)


# @app.post("/save_xml/{xml_file_name}")
# @catch_exceptions
# async def analyze_and_save_xml(file: UploadFile = File(...), xml_file_name: str | None = None, fast: bool = Form(False)):
#     if fast:
#         return await run_in_threadpool(analyze_pdf_fast, file.file.read(), xml_file_name)
#     return await run_in_threadpool(analyze_pdf, file.file.read(), xml_file_name)


# @app.get("/get_xml/{xml_file_name}", response_class=PlainTextResponse)
# @catch_exceptions
# async def get_xml_by_name(xml_file_name: str):
#     return await run_in_threadpool(get_xml, xml_file_name)


# @app.post("/toc")
# @catch_exceptions
# async def get_toc_endpoint(file: UploadFile = File(...), fast: bool = Form(False)):
#     return await run_in_threadpool(get_toc, file, fast)


# @app.post("/text")
# @catch_exceptions
# async def get_text_endpoint(file: UploadFile = File(...), fast: bool = Form(False), types: str = Form("all")):
#     return await run_in_threadpool(get_text_extraction, file, fast, types)


# @app.get("/readiness")
# def readiness():
#     return {"status": "ready"}


# @app.get("/")
# async def info():
#     return sys.version + " Using GPU: " + str(torch.cuda.is_available())


# @app.get("/error")
# async def error():
#     raise FileNotFoundError("This is a test error from the error endpoint")


# @app.post("/")
# @catch_exceptions
# async def run(file: UploadFile = File(...), fast: bool = Form(False), density: int = Form(72), extension: str = Form("jpeg")):
#     async with processing_lock:
#         if fast:
#             return await run_in_threadpool(analyze_pdf_fast, file.file.read(), "")

#         return await run_in_threadpool(analyze_pdf, file.file.read(), "", density, extension)

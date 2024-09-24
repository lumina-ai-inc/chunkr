import sys
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse

from catch_exceptions import catch_exceptions
from configuration import service_logger
from pdf_layout_analysis.get_xml import get_xml
from pdf_layout_analysis.run_pdf_layout_analysis import analyze_pdf
from pdf_layout_analysis.run_pdf_layout_analysis_fast import analyze_pdf_fast
from text_extraction.get_text_extraction import get_text_extraction
from toc.get_toc import get_toc

service_logger.info(f"Is PyTorch using GPU: {torch.cuda.is_available()}")

app = FastAPI()

@app.get("/readiness")
def readiness():
    return {"status": "ready"}


@app.get("/")
async def info():
    return sys.version + " Using GPU: " + str(torch.cuda.is_available())


@app.get("/error")
async def error():
    raise FileNotFoundError("This is a test error from the error endpoint")


from asyncio import Lock

processing_lock = Lock()

@app.post("/")
@catch_exceptions
async def run(file: UploadFile = File(...), fast: bool = Form(False), density: int = Form(72), extension: str = Form("jpeg")):
    async with processing_lock:
        if fast:
            return analyze_pdf_fast(file.file.read(), "")

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
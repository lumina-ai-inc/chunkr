from robyn import Robyn, Request
from tempfile import NamedTemporaryFile

app = Robyn(__file__)

@app.post("/ocr")
async def perform_ocr(request: Request):
    result = ""
    return result
from robyn import Robyn, Request
from tempfile import NamedTemporaryFile

app = Robyn(__file__)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/ocr")
async def perform_ocr(request: Request):
    result = ""
    return result

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app.start(host="0.0.0.0", port=port)

import asyncio
import argparse
from robyn import Robyn, Request, Response
from FlagEmbedding import FlagReranker
import gc
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, validator
from typing import List, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

load_dotenv(override=True)

app = Robyn(__file__)

reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float

class OCRResult(BaseModel):
    bbox: BoundingBox
    text: str
    confidence: Optional[float]

class Segment(BaseModel):
    segment_id: str
    bbox: BoundingBox
    page_number: int
    page_width: float
    page_height: float
    content: str
    segment_type: str
    ocr: Optional[List[OCRResult]]
    image: Optional[str]
    html: Optional[str]
    markdown: Optional[str]

class RerankRequest(BaseModel):
    query: str
    segments: List[Segment]

    @validator('segments', pre=True)
    def parse_segments(cls, v):
        if isinstance(v, str):
            try:
                segments_list = json.loads(v)
                if not isinstance(segments_list, list):
                    raise ValueError("segments must be a list")
                return segments_list
            except json.JSONDecodeError:
                raise ValueError("segments must be a valid JSON list")
        return v

class BatchRerankRequest(BaseModel):
    queries: List[str]
    segments: List[Segment]

    @validator('queries', pre=True)
    def parse_queries(cls, v):
        if isinstance(v, str):
            try:
                queries_list = json.loads(v)
                if not isinstance(queries_list, list):
                    raise ValueError("queries must be a list")
                return queries_list
            except json.JSONDecodeError:
                raise ValueError("queries must be a valid JSON list")
        return v

    @validator('segments', pre=True)
    def parse_segments(cls, v):
        if isinstance(v, str):
            try:
                segments_list = json.loads(v)
                if not isinstance(segments_list, list):
                    raise ValueError("segments must be a list")
                return segments_list
            except json.JSONDecodeError:
                raise ValueError("segments must be a valid JSON list")
        return v

@app.post("/rerank")
async def rerank_handler(request: Request):
    try:
        data = await request.json()  # Ensure asynchronous handling
        logging.info(f"Received /rerank data: {data}")

        rerank_request = RerankRequest(**data)
        query = rerank_request.query
        segments = rerank_request.segments

        passages = [segment.markdown for segment in segments if segment.markdown]

        scores = reranker.compute_score([[query, passage] for passage in passages])

        ranked_results = sorted(zip(segments, scores), key=lambda x: x[1], reverse=True)

        # Serialize Segment instances to dictionaries
        response_body = {"ranked_results": [segment.dict() for segment, score in ranked_results]}
        response_status = 200
        response_headers = {"Content-Type": "application/json"}

        return response_body, response_status, response_headers

    except ValidationError as ve:
        logging.error(f"Pydantic ValidationError in /rerank: {ve}")
        response_body = {"error": ve.errors()}
        response_status = 400
        response_headers = {"Content-Type": "application/json"}
        return response_body, response_status, response_headers
    except Exception as e:
        logging.error(f"Unexpected error in /rerank: {e}")
        response_body = {"error": "Internal Server Error"}
        response_status = 500
        response_headers = {"Content-Type": "application/json"}
        return response_body, response_status, response_headers

@app.post("/batch_rerank")
async def batch_rerank_handler(request: Request):
    try:
        data = request.json()  # Ensure asynchronous handling
        logging.info(f"Received /batch_rerank data: {data}")

        batch_rerank_request = BatchRerankRequest(**data)
        queries = batch_rerank_request.queries
        segments = batch_rerank_request.segments

        results = {}
        for query in queries:
            passages = [segment.markdown for segment in segments if segment.markdown]
            
            scores = reranker.compute_score([[query, passage] for passage in passages])
            
            ranked_results = sorted(zip(segments, scores), key=lambda x: x[1], reverse=True)
            
            # Serialize Segment instances to dictionaries
            results[query] = [segment.dict() for segment, score in ranked_results]
        
        response_body = {"batch_ranked_results": results}
        return response_body

    except ValidationError as ve:
        logging.error(f"Pydantic ValidationError in /batch_rerank: {ve}")
        response_body = {"error": ve.errors()}
        return response_body
    except Exception as e:
        logging.error(f"Unexpected error in /batch_rerank: {e}")
        response_body = {"error": "Internal Server Error"}
        return response_body

def test_reranker():
    test_query = "vegetables"
    test_segments = [
        Segment(
            segment_id="1",
            bbox=BoundingBox(left=0, top=0, width=100, height=100),
            page_number=1,
            page_width=800.0,
            page_height=600.0,
            content="Apple is a sweet fruit that grows on trees.",
            segment_type="text",
            ocr=None,
            image=None,
            html=None,
            markdown="Apple is a sweet fruit that grows on trees."
        ),
        Segment(
            segment_id="2",
            bbox=BoundingBox(left=0, top=0, width=100, height=100),
            page_number=1,
            page_width=800.0,
            page_height=600.0,
            content="Bananas are rich in potassium and are a popular snack.",
            segment_type="text",
            ocr=None,
            image=None,
            html=None,
            markdown="Bananas are rich in potassium and are a popular snack."
        ),
        Segment(
            segment_id="3",
            bbox=BoundingBox(left=0, top=0, width=100, height=100),
            page_number=1,
            page_width=800.0,
            page_height=600.0,
            content="Carrots are root vegetables that are high in beta-carotene.",
            segment_type="text",
            ocr=None,
            image=None,
            html=None,
            markdown="Carrots are root vegetables that are high in beta-carotene."
        ),
        Segment(
            segment_id="4",
            bbox=BoundingBox(left=0, top=0, width=100, height=100),
            page_number=1,
            page_width=800.0,
            page_height=600.0,
            content="Eggplants are versatile vegetables used in various cuisines around the world.",
            segment_type="text",
            ocr=None,
            image=None,
            html=None,
            markdown="Eggplants are versatile vegetables used in various cuisines around the world."
        ),
        Segment(
            segment_id="5",
            bbox=BoundingBox(left=0, top=0, width=100, height=100),
            page_number=1,
            page_width=800.0,
            page_height=600.0,
            content="Dates are sweet fruits that are often dried and used in desserts.",
            segment_type="text",
            ocr=None,
            image=None,
            html=None,
            markdown="Dates are sweet fruits that are often dried and used in desserts."
        )
    ]
    passages = [segment.markdown for segment in test_segments if segment.markdown]
    scores = reranker.compute_score([[test_query, passage] for passage in passages])
    ranked_results = sorted(zip(test_segments, scores), key=lambda x: x[1], reverse=True)
    
    print("\nTest Query:", test_query)
    print("\nRanked Segments:")
    for segment, score in ranked_results:
        print(f"\nScore: {score:.4f}")
        print(f"Segment ID: {segment.segment_id}")
        print(f"Content: {segment.content}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the reranker service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8011, help="Port to run the server on")
    args = parser.parse_args()

    test_reranker()

    try:
        app.start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("Server shutdown initiated by user.")
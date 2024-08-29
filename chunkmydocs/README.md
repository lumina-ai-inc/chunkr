# Chunk My Docs

We're Lumina. We've built a search engine that's 5x more relevant than Google Scholar. You can check us out at https://lumina.sh. We achieved this by bringing state of the art search technology (the best in dense and sparse vector embeddings) to academic research. 

While search is one problem, sourcing high quality data is another. We needed to process millions of PDFs in house to build Lumina, and we found out that existing solutions to extract structured information from PDFs were too slow and too expensive ($$ per page). 

Chunk my docs provides a self-hostable solution that leverages state-of-the-art (SOTA) vision models for segment extraction and OCR, unifying the output through a Rust Actix server. This setup allows you to process PDFs and extract segments at an impressive speed of approximately 60 pages per second on a single NVIDIA L4 instance, offering a cost-effective and scalable solution for high-accuracy bounding box segment extraction and OCR. This solution has models that accomodate for both GPU and CPU environments. 

## Features:

- Bounding Box extraction and Element Detection
- Markdown Generation

## Local Dev Guide

Our setup runs the Rust actix-web server locally on metal and everything else in Docker. pdla (pdf-document-layout-analysis) is meant to run on GPU so you may find it to be slow when running locally on CPU.

### 1\. Setup ENV's

`cp .env.docker-compose .env`

`cp .env.chunkmydocs ./chunkmydocs/.env`

`cp .env.pyscripts ./pyscripts/.env`

### 2\. Run the things

`docker compose up -d`

Then, run the server and task worker:

```
cd chunkmydocs
cargo run
cargo run --bin task-processor
```

### 3\. Get local API key

Run the following curl script to get an API key:

```
curl -X POST http://localhost:8000/api_key \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "example_user_id",
    "email": "givme@apikey.com",
    "access_level": "OWNER",
    "expires_at": "2023-12-31T23:59:59Z",
    "initial_usage": 0,
    "usage_limit": 100000,
    "usage_type": "FREE",
    "service_type": "EXTRACTION"
  }'
```

Copy the resulting key.

Paste the key into `pyscripts/.env` as the value for `INGEST_SERVER__API_KEY`.

### 4\. Test that things are working

`cd pyscripts && mkdir input && mkdir output`

Then, put some PDF into the `./pyscripts/input` folder. I recommend [Justice Department Sues Apple for Monopolizing Smartphone Markets](https://www.justice.gov/opa/media/1344546/dl?inline).

`cd pyscripts && python3 main.py`

Once that finishes, you can view the resulting chunks in `pyscripts/output/{file_name}-Fast/bounding_boxes.json`.

## Acknowledgements 

Special shout out to Trieve (https://trieve.ai) for contributing and helping us make this project source available. Trieve is all-in-one infrastructure for building hybrid vector search, recommendations, and RAG. Trieve is also source available (https://github.com/devflowinc/trieve), and is used extensively on lumina.sh.

## Roadmap

- integrate with Trieve
- add support for Grobid
- make a diagram
- explain how insanely awesome [RRQ](https://github.com/lumina-ai-inc/resilient-redis-queue) is
- Kube deploy guide similar to [trieve/self-hosting.md](https://github.com/devflowinc/trieve/blob/main/self-hosting.md)

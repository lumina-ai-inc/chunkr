# chunk-my-docs

lumina x trieve cookup. SOTA pdf extraction

woooo lfg

HO w to chukn a PDF??

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

## Roadmap

- integrate with Trieve
- add support for Grobid
- make a diagram
- explain how insanely awesome [RRQ](https://github.com/lumina-ai-inc/resilient-redis-queue) is
- Kube deploy guide similar to [trieve/self-hosting.md](https://github.com/devflowinc/trieve/blob/main/self-hosting.md)

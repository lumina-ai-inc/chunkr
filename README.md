### Status Updates (Only for hosted API on www.chunkr.ai)
1. We have temporarily switched to Textract for OCR from PaddleOCR. Textract is provided for free until we resolve PaddleOCR issues. Textract occasionally misses tables that PaddleOCR wouldn't. For self-deploys, you can still set PaddleOCR as your OCR strategy in the task service .env variables.
2. We are still experiencing extremely high loads, which have affected throughputs. We're working hard to get ingestion speeds back to our standard. 

# Chunkr

We're Lumina. We've built a search engine that's five times more relevant than Google Scholar. You can check us out at [lumina.sh](https://www.lumina.sh). We achieved this by bringing state-of-the-art search technology (the best in dense and sparse vector embeddings) to academic research. 

While search is one problem, sourcing high-quality data is another. We needed to process millions of PDFs in-house to build Lumina, and we found that existing solutions to extract structured information from PDFs were too slow and too expensive ($$ per page). 

Chunk my docs provides a self-hostable solution that leverages state-of-the-art (SOTA) vision models for segment extraction and OCR, unifying the output through a Rust Actix server. This setup allows you to process PDFs and extract segments at an impressive speed of approximately 5 pages per second on a single NVIDIA L4 instance, offering a cost-effective and scalable solution for high-accuracy bounding box segment extraction and OCR. This solution has models that accommodate both GPU and CPU environments. Try the UI on [chunkr.ai](https://www.chunkr.ai)!

## Docs

https://docs.chunkr.ai/introduction

## (Super) Quick Start

1. Go to [chunkr.ai](https://www.chunkr.ai) 
2. Make an account and copy your API key
3. Create a task:
   ```bash
   curl -X POST https://api.chunkr.ai/api/v1/task \
      -H "Content-Type: multipart/form-data" \
      -H "Authorization: ${YOUR_API_KEY}" \
      -F "file=@/path/to/your/file" \
      -F "model=HighQuality" \
      -F "target_chunk_length=512" \
      -F "ocr_strategy=Auto"
   ```
4. Poll your created task:
    ```bash
   curl -X GET https://api.chunkr.ai/api/v1/task/${TASK_ID} \
      -H "Authorization: ${YOUR_API_KEY}"
   ```

## Self Deployments

1. You'll need K8s and docker.
2. Follow the steps in `self-deployment.md`

## Licensing

This project is dual-licensed:

1. [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)
2. Commercial License

To use Chunkr without complying with the AGPL-3.0 license terms you can [contact us](mailto:mehul@lumina.sh) or visit our [website](https://chunkr.ai).

## Want to talk to a founder?
https://cal.com/mehulc/30min

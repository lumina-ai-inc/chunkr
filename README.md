<br />
<div align="center">
  <a href="https://github.com/lumina-ai-inc/chunkr">
    <img src="images/logo.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Chunkr | Open Source Document Intelligence API</h3>

  <p align="center">
    Production-ready API service for document layout analysis, OCR, and semantic chunking. Convert PDFs, PPTs, Word docs & images into RAG/LLM-ready chunks.
    <br />
    <b>Layout Analysis</b> | <b>OCR + Bounding Boxes</b> | <b>Structured HTML and markdown</b> | <b>VLM Processing controls</b>
    <br />
    <br />
    <a href="https://www.chunkr.ai">Try it out!</a>
    ·
    <a href="https://github.com/lumina-ai-inc/chunkr/issues/new">Report Bug</a>
    ·
    <a href="#connect-with-us">Contact</a>
  </p>
</div>


<div align="center">
  <a href="https://www.chunkr.ai" width="1200" height="630">
    <img src="https://chunkr.ai/og-image.png" style="bor">
  </a>
</div>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [(Super) Quick Start](#super-quick-start)
- [Documentation](#documentation)
- [Self-Hosted Deployment Options](#self-hosted-deployment-options)
  - [Quick Start with Docker Compose](#quick-start-with-docker-compose)
  - [Deployment with Kubernetes](#deployment-with-kubernetes)
- [Licensing](#licensing)
- [Connect With Us](#connect-with-us)

## (Super) Quick Start

1. Go to [chunkr.ai](https://www.chunkr.ai) 
2. Make an account and copy your API key
3. Install our Python SDK:
   ```bash
   pip install chunkr-ai
   ```
4. Use the SDK to process your documents:
   ```python
   from chunkr_ai import Chunkr

   # Initialize with your API key from chunkr.ai
   chunkr = Chunkr(api_key="your_api_key")

   # Upload a document (URL or local file path)
   url = "https://chunkr-web.s3.us-east-1.amazonaws.com/landing_page/input/science.pdf"
   task = chunkr.upload(url)

   # Export results in various formats
   task.html(output_file="output.html")
   task.markdown(output_file="output.md")
   task.content(output_file="output.txt")
   task.json(output_file="output.json")

   # Clean up
   chunkr.close()
   ```

## Documentation

Visit our [docs](https://docs.chunkr.ai) for more information and examples.

## Self-Hosted Deployment Options

### Quick Start with Docker Compose
1. Prerequisites:
   - [Docker and Docker Compose](https://docs.docker.com/get-docker/)
   - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support, optional)

2. Clone the repo:
```bash
git clone https://github.com/lumina-ai-inc/chunkr
cd chunkr
```

3. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Configure your environment variables
# Required: LLM_KEY as your OpenAI API key
```

4. Start the services:
   
With GPU (recommended):
```bash
docker compose up -d
```

Only CPU (under development):
```bash
docker compose -f compose-cpu.yaml up -d
```

1. Access the services:
   - Web UI: `http://localhost:5173`
   - API: `http://localhost:8000`

> **Note**: The default configuration (`docker compose up -d`) requires an NVIDIA CUDA GPU. For systems without a GPU, use the CPU deployment option.

6. Stop the services when done:
```bash
docker compose down
```

### Deployment with Kubernetes
For production environments, we provide a Helm chart and detailed deployment instructions:
1. See our detailed guide at [`kube/README.md`](kube/README.md)
2. Includes configurations for high availability and scaling

For enterprise support and deployment assistance, [contact us](mailto:mehul@lumina.sh).

## Licensing

This project is dual-licensed:

1. [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)
2. Commercial License

To use Chunkr without complying with the AGPL-3.0 license terms you can [contact us](mailto:mehul@lumina.sh) or visit our [website](https://chunkr.ai).

## Connect With Us
- 📧 Email: [mehul@lumina.sh](mailto:mehul@lumina.sh)
- 📅 Schedule a call: [Book a 30-minute meeting](https://cal.com/mehulc/30min)
- 🌐 Visit our website: [chunkr.ai](https://chunkr.ai)

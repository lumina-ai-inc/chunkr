<br />
<div align="center">
  <a href="https://github.com/lumina-ai-inc/chunkr">
    <img src="images/logo.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Chunkr | Open Source Document Intelligence API</h3>

  <p align="center">
    Production-ready API service for document layout analysis, OCR, and semantic chunking.<br />Convert PDFs, PPTs, Word docs & images into RAG/LLM-ready chunks.
    <br /><br />
    <b>Layout Analysis</b> | <b>OCR + Bounding Boxes</b> | <b>Structured HTML and markdown</b> | <b>VLM Processing controls</b>
    <br />
    <br />
    <a href="https://www.chunkr.ai"><img src="https://img.shields.io/badge/Try_it_out-chunkr.ai-blue?style=flat&logo=rocket&height=20" alt="Try it out" height="20"></a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://github.com/lumina-ai-inc/chunkr/issues/new"><img src="https://img.shields.io/badge/Report_Bug-GitHub_Issues-red?style=flat&logo=github&height=20" alt="Report Bug" height="20"></a>
    &nbsp;&nbsp;&nbsp;
    <a href="#connect-with-us"><img src="https://img.shields.io/badge/Contact-Get_in_Touch-green?style=flat&logo=mail&height=20" alt="Contact" height="20"></a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://discord.gg/XzKWFByKzW"><img src="https://img.shields.io/badge/Discord-Join_Community-5865F2?style=flat&logo=discord&logoColor=white&height=20" alt="Discord" height="20"></a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://deepwiki.com/lumina-ai-inc/chunkr"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
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
    - [HTTPS Setup for Docker Compose](#https-setup-for-docker-compose)
  - [Deployment with Kubernetes](#deployment-with-kubernetes)
- [LLM Configuration](#llm-configuration)
  - [Using models.yaml (Recommended)](#using-modelsyaml-recommended)
  - [Using environment variables (Basic)](#using-environment-variables-basic)
  - [Common LLM API Providers](#common-llm-api-providers)
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
html = task.html(output_file="output.html")
markdown = task.markdown(output_file="output.md")
content = task.content(output_file="output.txt")
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

# Configure your llm models
cp models.example.yaml models.yaml
```

**Note on Python Dependencies**: For Python services, `pyproject.toml` is the single source of truth for dependencies. `requirements.txt` is not used.

For more information on how to set up LLMs, see [here](#llm-configuration).

4. Start the services:
```bash
# For GPU deployment, use the following command:
docker compose up -d

# For CPU deployment, use the following command:
docker compose -f compose-cpu.yaml up -d

# For Mac ARM architecture (eg. M2, M3 etc.) deployment, use the following command:
docker compose -f compose-cpu.yaml -f compose-mac.yaml up -d
```

5. Access the services:
   - Web UI: `http://localhost:5173`
   - API: `http://localhost:8000`

6. Stop the services when done:
```bash
# For GPU deployment, use the following command:
docker compose down

# For CPU deployment, use the following command:
docker compose -f compose-cpu.yaml down

# For Mac ARM architecture (eg. M2, M3 etc.) deployment, use the following command:
docker compose -f compose-cpu.yaml -f compose-mac.yaml down
```

#### HTTPS Setup for Docker Compose

This section explains how to set up HTTPS using a self signed certificate with Docker Compose when hosting Chunkr on a VM. This allows you to access the web UI, API, Keycloak (authentication service) and MinIO (object storage service) over HTTPS.

1. Generate a self-signed certificate:
```bash
# Create a certs directory
mkdir certs

# Generate the certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout certs/nginx.key -out certs/nginx.crt -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

2. Update the .env file with your VM's IP address:
> **Important**: Replace all instances of "localhost" with your VM's actual IP address. Note that you must use "https://" instead of "http://" and the ports are different from the HTTP setup (No port for web, 8444 for API, 8443 for Keycloak, 9100 for MinIO):
```bash
AWS__PRESIGNED_URL_ENDPOINT=https://your_vm_ip_address:9100
WORKER__SERVER_URL=https://your_vm_ip_address:8444
VITE_API_URL=https://your_vm_ip_address:8444
VITE_KEYCLOAK_POST_LOGOUT_REDIRECT_URI=https://your_vm_ip_address
VITE_KEYCLOAK_REDIRECT_URI=https://your_vm_ip_address
VITE_KEYCLOAK_URL=https://your_vm_ip_address:8443
```

1. Start the services:
```bash
# For GPU deployment, use the following command:
docker compose --profile proxy up -d

# For CPU deployment, use the following command:
docker compose -f compose-cpu.yaml --profile proxy up -d

# For Mac ARM architecture (eg. M2, M3 etc.) deployment, use the following command:
docker compose -f compose-cpu.yaml -f compose-mac.yaml --profile proxy up -d
```

4. Access the services:
   - Web UI: `https://your_vm_ip_address`
   - API: `https://your_vm_ip_address:8444`

5. Stop the services when done:
```bash
# For GPU deployment, use the following command:
docker compose --profile proxy down

# For CPU deployment, use the following command:
docker compose -f compose-cpu.yaml --profile proxy down

# For Mac ARM architecture (eg. M2, M3 etc.) deployment, use the following command:
docker compose -f compose-cpu.yaml -f compose-mac.yaml --profile proxy down
```

### Deployment with Kubernetes

For production environments, we provide a Helm chart and detailed deployment instructions:
1. See our detailed guide at [`kube/README.md`](kube/README.md)
2. Includes configurations for high availability and scaling

For enterprise support and deployment assistance, [contact us].

## LLM Configuration

Chunkr supports two ways to configure LLMs:

1. **models.yaml file**: Advanced configuration for multiple LLMs with additional options
2. **Environment variables**: Simple configuration for a single LLM

### Using models.yaml (Recommended)

For more flexible configuration with multiple models, default/fallback options, and rate limits:

1. Copy the example file to create your configuration:
```bash
cp models.example.yaml models.yaml
```

2. Edit the models.yaml file with your configuration. Example:
```yaml
models:
  - id: gpt-4o
    model: gpt-4o
    provider_url: https://api.openai.com/v1/chat/completions
    api_key: "your_openai_api_key_here"
    default: true
    rate-limit: 200 # requests per minute - optional
```

Benefits of using models.yaml:
- Configure multiple LLM providers simultaneously
- Set default and fallback models
- Add distributed rate limits per model
- Reference models by ID in API requests (see docs for more info)

>Read the `models.example.yaml` file for more information on the available options.

### Using environment variables (Basic)

You can use any OpenAI API compatible endpoint by setting the following variables in your .env file:
``` 
LLM__KEY:
LLM__MODEL:
LLM__URL:
```

### Common LLM API Providers

Below is a table of common LLM providers and their configuration details to get you started:

| Provider         | API URL                                                                  | Documentation                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| OpenAI           | https://api.openai.com/v1/chat/completions                               | [OpenAI Docs](https://platform.openai.com/docs)                                                                                        |
| Google AI Studio | https://generativelanguage.googleapis.com/v1beta/openai/chat/completions | [Google AI Docs](https://ai.google.dev/gemini-api/docs/openai)                                                                         |
| OpenRouter       | https://openrouter.ai/api/v1/chat/completions                            | [OpenRouter Models](https://openrouter.ai/models)                                                                                      |
| Self-Hosted      | http://localhost:8000/v1                                                 | [VLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) or [Ollama](https://ollama.com/blog/openai-compatibility) |

## Licensing

The core of this project is dual-licensed:

1. [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)
2. Commercial License

To use Chunkr without complying with the AGPL-3.0 license terms you can [contact us](mailto:mehul@chunkr.ai) or visit our [website](https://chunkr.ai).

## Connect With Us
- 📧 Email: mehul@chunkr.ai
- 📅 Schedule a call: [Book a 30-minute meeting](https://cal.com/mehulc/30min)
- 🌐 Visit our website: [chunkr.ai](https://chunkr.ai)

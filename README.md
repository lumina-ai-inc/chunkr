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
- [OpenSource vs Commercial API vs Enterprise](#opensource-vs-commercial-api-vs-enterprise)
- [Self-Hosted Deployment Options](#self-hosted-deployment-options)
  - [Quick Start with Docker Compose](#quick-start-with-docker-compose)
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

## OpenSource vs Commercial API vs Enterprise

| Feature | Open Source | Commercial API | Enterprise |
|---------|-------------|----------------|------------|
| **Perfect for** | Development & testing | Production applications | Large-scale/High security deployments|
| **Layout Analysis** | Basic models | Advanced models | Advanced + custom-tuned |
| **OCR Accuracy** | Standard models | Premium models | Premium + domain-tuned |
| **VLM Processing** | Basic vision models | Enhanced VLM models | Enhanced + custom training |
| **Excel Support** | ❌ | ✅ Native parser | ✅ Native parser |
| **Document Types** | PDF, PPT, Word, Images | PDF, PPT, Word, Images, Excel | PDF, PPT, Word, Images, Excel |
| **Infrastructure** | Self-hosted | Fully managed | Fully managed  |
| **Data Control** | Full (on your servers) | Chunkr-hosted | Full (on your infrastructure) or Chunkr-hosted |
| **Deployment** | Docker self-managed | Cloud API | On-premises/Private cloud |
| **Support** | Discord community | Priority email + community | 24/7 dedicated founder support |
| **Updates** | Manual (open source) | Automatic | Custom deployment schedule |
| **Customization** | Code-level modifications | API configuration | Full model tuning + custom SLAs |
| **Migration Support** | Community resources | Documentation + email | Dedicated migration team |

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
- 📧 Email: [mehul@chunkr.ai](mailto:mehul@chunkr.ai)
- 📅 Schedule a call: [Book a 30-minute meeting](https://cal.com/mehulc/30min)
- 🌐 Visit our website: [chunkr.ai](https://chunkr.ai)

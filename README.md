<br />
<div align="center">
  <a href="https://github.com/lumina-ai-inc/chunkr">
    <img src="images/logo.svg" alt="Chunkr Logo" width="80" height="80">
  </a>

<h3 align="center">Chunkr | Open Source Document Intelligence API</h3>

  <p align="center">
    Production-ready service for document layout analysis, OCR, and semantic chunking.<br />
    Convert PDFs, PPTs, Word docs & images into RAG/LLM-ready chunks.
    <br /><br />
    <b>Layout Analysis</b> | <b>OCR + Bounding Boxes</b> | <b>Structured HTML & Markdown</b> | <b>Vision-Language Model Processing</b>
    <br /><br />
    👉 <b>Note:</b> The <a href="https://github.com/lumina-ai-inc/chunkr">open-source AGPL version</a> is **different** from our fully managed <a href="https://www.chunkr.ai">Cloud API</a>.  
    The open-source release uses community/open-source models, while the Cloud API runs **proprietary in-house models** for higher accuracy, speed, and enterprise reliability.
    <br /><br />
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
    <img src="https://chunkr.ai/og-image.png" alt="Chunkr Cloud API">
  </a>
</div>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [(Super) Quick Start](#super-quick-start)
- [Documentation](#documentation)
- [Open Source vs Cloud API vs Enterprise](#open-source-vs-cloud-api-vs-enterprise)
- [Quick Start with Docker Compose](#quick-start-with-docker-compose)
- [LLM Configuration](#llm-configuration)
  - [Using models.yaml (Recommended)](#using-modelsyaml-recommended)
  - [Using environment variables (Basic)](#using-environment-variables-basic)
  - [Common LLM API Providers](#common-llm-api-providers)
- [Licensing](#licensing)
- [Connect With Us](#connect-with-us)

## Open Source vs Cloud API vs Enterprise

| Feature | Open Source (AGPL) | Cloud API (chunkr.ai) | Enterprise |
|---------|--------------------|------------------------|------------|
| **Perfect for** | Development & testing | Production workloads | Large-scale / High-security |
| **Layout Analysis** | Uses open-source models | Proprietary in-house models | In-house + custom-tuned |
| **OCR Accuracy** | Community OCR engines | Optimized OCR stack | Optimized + domain-tuned |
| **VLM Processing** | Basic open VLMs | Enhanced proprietary VLMs | Custom fine-tunes |
| **Excel Support** | ❌ | ✅ Native parser | ✅ Native parser |
| **Document Types** | PDF, PPT, Word, Images | PDF, PPT, Word, Images, Excel | PDF, PPT, Word, Images, Excel |
| **Infrastructure** | Self-hosted | Fully managed cloud | Managed / On-prem |
| **Support** | Discord community | Dedicated support | Dedicated founding team |
| **Migration Support** | Community-driven | Docs + email | Dedicated migration team |

---

The **open-source release** is ideal if you want transparency, local hosting, or to experiment with Chunkr’s pipeline.  
For **best performance, production reliability, and access to in-house models**, we recommend the <a href="https://www.chunkr.ai">Chunkr Cloud API</a>.  
For **high-security or regulated industries**, our **Enterprise edition** offers on-prem or VPC deployments.


## Quick Start with Docker Compose

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
# For GPU deployment:
docker compose up -d

# For CPU-only deployment:
docker compose -f compose.yaml -f compose.cpu.yaml up -d

# For Mac ARM architecture (M1, M2, M3, etc.):
docker compose -f compose.yaml -f compose.cpu.yaml -f compose.mac.yaml up -d
```

5. Access the services:
   - Web UI: `http://localhost:5173`
   - API: `http://localhost:8000`

6. Stop the services when done:
```bash
# For GPU deployment:
docker compose down

# For CPU-only deployment:
docker compose -f compose.yaml -f compose.cpu.yaml down

# For Mac ARM architecture (M1, M2, M3, etc.):
docker compose -f compose.yaml -f compose.cpu.yaml -f compose.mac.yaml down
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

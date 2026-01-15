<br />
<div align="center">
  <a href="https://github.com/buildorin/data-extract">
    <img src="images/logo.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Orin | AI-Powered Real Estate Deal Analysis</h3>

  <p align="center">
    SaaS platform for real estate fund managers to analyze deals, extract insights, and generate investor-ready packages.<br />Turn rental portfolios and property documents into funding opportunities.
    <br /><br />
    <b>AI Agents</b> | <b>Document Processing</b> | <b>Underwriting Analysis</b> | <b>Investor Packages</b>
    <br />
    <br />
    <a href="https://www.chunkr.ai"><img src="https://img.shields.io/badge/Try_it_out-chunkr.ai-blue?style=flat&logo=rocket&height=20" alt="Try it out" height="20"></a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://github.com/buildorin/data-extract/issues/new"><img src="https://img.shields.io/badge/Report_Bug-GitHub_Issues-red?style=flat&logo=github&height=20" alt="Report Bug" height="20"></a>
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
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [GCP Deployment](#gcp-deployment)
- [LLM Configuration](#llm-configuration)
  - [Using models.yaml (Recommended)](#using-modelsyaml-recommended)
  - [Using environment variables (Basic)](#using-environment-variables-basic)
  - [Common LLM API Providers](#common-llm-api-providers)
- [Licensing](#licensing)
- [Connect With Us](#connect-with-us)

## Quick Start

Visit the Orin platform at your deployment URL to get started analyzing real estate deals.

## Development Setup

### Prerequisites
- [Docker and Docker Compose](https://docs.docker.com/get-docker/)
- Node.js 18+ (for frontend development)
- Rust 1.70+ (for backend development)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/buildorin/data-extract
cd data-extract
```

2. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Configure your LLM models
cp models.example.yaml models.yaml
```

3. Start all services:
```bash
make start
```

4. Access the development environment:
   - Web UI: `http://localhost:5173`
   - API: `http://localhost:8000`
   - Keycloak: `http://localhost:8080`
   - MinIO: `http://localhost:9001`
   - Qdrant: `http://localhost:6333`

### Common Commands

**Daily Development:**
```bash
make start-no-build  # Fastest - use existing builds
make logs            # View logs
make stop            # Stop services
```

**After Code Changes:**
```bash
make build-backend   # Rebuild backend only
make build-frontend  # Rebuild frontend only
make build-all       # Rebuild everything
make restart         # Restart services
```

**Maintenance:**
```bash
make status          # Check service status
make clean           # Clean up everything
```

**Note:** On Mac, services automatically run in CPU mode (no GPU required). See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for details.

## GCP Deployment

### Prerequisites
- Google Cloud Project with billing enabled
- `gcloud` CLI installed and configured
- Terraform installed (optional, for IaC)

### Deploy to GCP

1. Set up your GCP project:
```bash
export PROJECT_ID=your-gcp-project-id
gcloud config set project $PROJECT_ID
```

2. Deploy using the deployment script:
```bash
make deploy
```

Or manually:
```bash
./scripts/deploy-gcp.sh
```

### Architecture
The GCP deployment includes:
- Compute Engine VM with GPU for OCR/Segmentation workloads
- Containerized services (Postgres, Redis, MinIO, Qdrant, Keycloak)
- Cloud Load Balancer for HTTPS termination
- Cloud Storage for document backups
- Secret Manager for sensitive configuration

For detailed deployment instructions, see:
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Complete deployment guide
- [gcp/README.md](gcp/README.md) - GCP-specific details

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

The core of this project is licensed under:

1. [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)

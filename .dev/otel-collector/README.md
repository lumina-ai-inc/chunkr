# Local OpenTelemetry Collector for SignOz

This directory contains configuration for running a local OpenTelemetry collector that forwards telemetry data to SignOz Cloud.

## Setup

1. Copy the template to create your local config:
   ```bash
   cp otel-collector-config.template.yaml otel-collector-config.yaml
   ```

2. Edit `otel-collector-config.yaml` to add your SignOz ingestion key:
   ```yaml
   headers:
     "signoz-ingestion-key": "YOUR_SIGNOZ_INGESTION_KEY" # Replace with your key
   ```

3. Start the collector:
   ```bash
   docker compose up -d
   ```

4. Configure your application with these environment variables:
   ```bash
   # Required: OpenTelemetry endpoint
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
   
   # Optional: Configure service namespace and environment
   export OTEL__SERVICE_NAMESPACE=chunkr
   export OTEL__DEPLOYMENT_ENVIRONMENT=dev
   ```

## Configuration Options

The following environment variables are supported:

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP exporter endpoint |
| `OTEL__SERVICE_NAMESPACE` | `chunkr` | Service namespace |
| `OTEL__DEPLOYMENT_ENVIRONMENT` | `dev` | Deployment environment |

## Verification

1. Check if the collector is running:
   ```bash
   docker compose ps
   ```

2. View collector logs:
   ```bash
   docker compose logs -f
   ```

3. After starting your application, make a test request and verify that traces appear in both:
   - The collector logs (via the debug exporter)
   - The SignOz Cloud UI

## Important Notes

- The `otel-collector-config.yaml` file is excluded from git (.gitignore) to prevent accidental credential leakage
- You can change the SignOz region in the endpoint URL if needed (us/eu/in)
- The application automatically handles distributed tracing context propagation
- LLM error attributes are automatically extracted and included in spans
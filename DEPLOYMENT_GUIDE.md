# Orin Deployment Guide

This guide covers the simplified deployment process for the Orin SaaS platform.

## Overview

Orin has been simplified from a complex self-hosted system to a streamlined SaaS platform designed for Google Cloud Platform deployment. The architecture has been consolidated from 7+ Docker Compose variants to a single production configuration.

## Quick Start

### Development (Local Mac)

**Daily Development (Fastest):**
```bash
make start-no-build  # Quick start with existing builds
make logs            # View logs
make stop            # Stop services
```

**First Time Setup:**
```bash
make build-all       # Rebuild backend + frontend
make start           # Start all services
```

**After Code Changes:**
```bash
make build-backend   # Backend changes only
make build-frontend  # Frontend changes only
make restart         # Restart services
```

**Access Services:**
- Web UI: `http://localhost:5173`
- API: `http://localhost:8000`
- Keycloak: `http://localhost:8080`
- MinIO: `http://localhost:9001`
- Qdrant: `http://localhost:6333`

**Note:** On Mac, services run in CPU mode (no GPU). GPU requirements are automatically removed via `docker-compose.local.yaml`.

### Production Deployment (GCP)

```bash
# Deploy to GCP
make deploy
```

Or manually:
```bash
./scripts/deploy-gcp.sh
```

## Deployment Modes

Orin supports two deployment modes:

### Local Development (Mac)
- **Platform**: Mac (ARM64 or Intel)
- **GPU**: None (CPU mode via emulation)
- **Use Case**: Development, testing
- **Cost**: Free (runs locally)
- **Performance**: Slower (x86_64 emulation on Apple Silicon)
- **Command**: `make start`
- **Auto-detection**: Makefile automatically detects Mac and removes GPU requirements

**Services Running Locally:**
- Server (x2), Task Workers (x5), OCR Backend (x2, CPU), Segmentation (x2, CPU)
- PostgreSQL, Redis, MinIO, Qdrant, Keycloak

### Production (GCP)
- **Platform**: Linux (x86_64)
- **GPU**: NVIDIA T4
- **Use Case**: Production workloads
- **Cost**: ~$700-1,055/month
- **Performance**: Fast (native, GPU-enabled)
- **Command**: `make deploy`

**Services in Production:**
- Same as local, but GPU-enabled OCR/segmentation, auto-scaling, high availability

## Architecture Changes

### Before (Chunkr - Self-Hosted)
- 7 different Docker Compose configurations
- 480+ line dev-run.sh script
- 30 task worker replicas
- 6 segmentation replicas
- 3 OCR replicas
- Extensive self-hosting documentation
- Multiple deployment modes (GPU/CPU/Mac/Local)

### After (Orin - SaaS)
- Single production Docker Compose file
- Simple Makefile with 6 commands
- 5 task worker replicas (scalable)
- 2 segmentation replicas
- 2 OCR replicas
- GCP-focused deployment
- Consolidated configuration files

## Configuration

### Environment Configuration

Two primary configuration files:
- `config/development.yaml` - Local development
- `config/production.yaml` - Production deployment

Frontend environment files:
- `apps/web/.env.development` - Development settings
- `apps/web/.env.production` - Production settings

### GCP Configuration

Located in `gcp/` directory:
- `cloudbuild.yaml` - Cloud Build pipeline
- `terraform/main.tf` - Infrastructure as Code
- `terraform/variables.tf` - Terraform variables
- `README.md` - Detailed GCP deployment guide

## Deployment Process

### Step 1: Prerequisites

1. **Google Cloud Setup**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud auth configure-docker
   ```

2. **Enable Required APIs**
   - Compute Engine API
   - Container Registry API
   - Cloud Build API
   - Secret Manager API
   - Cloud Storage API

### Step 2: Configuration

1. **Set Environment Variables**
   
   For production, use GCP Secret Manager:
   ```bash
   # Database password
   gcloud secrets create orin-db-password --data-file=-
   
   # API keys
   gcloud secrets create orin-api-keys --data-file=-
   ```

2. **Configure LLM Models**
   
   Update `models.yaml` with your LLM provider settings.

### Step 3: Infrastructure Deployment (Optional - Using Terraform)

```bash
cd gcp/terraform

# Initialize
terraform init

# Plan
terraform plan -var="project_id=YOUR_PROJECT_ID"

# Apply
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

### Step 4: Application Deployment

```bash
# From project root
make deploy
```

Or manually:
```bash
./scripts/deploy-gcp.sh
```

This will:
1. Build all container images
2. Push to Google Container Registry
3. Deploy to Compute Engine
4. Run database migrations
5. Perform health checks

## Service Architecture

```
Production Stack:
├── Server (x2 replicas)
├── Task Workers (x5 replicas)
├── OCR Backend (x2 replicas, GPU)
├── Segmentation Backend (x2 replicas, GPU)
├── Web Frontend
├── PostgreSQL (containerized)
├── Redis (containerized)
├── MinIO (containerized)
├── Qdrant (containerized)
└── Keycloak (containerized)
```

## Scaling

### Vertical Scaling
Increase VM size in `gcp/terraform/main.tf`:
```hcl
machine_type = "n1-standard-16"  # Increase as needed
```

### Horizontal Scaling
Increase replicas in `docker-compose.production.yaml`:
```yaml
task:
  deploy:
    replicas: 10  # Scale based on load
```

## Monitoring

### View Logs
```bash
# All services
make logs

# Specific service
docker logs -f orin-server-1

# GCP Compute Engine logs
gcloud compute ssh orin-production --zone=us-central1-a
```

### Health Checks
```bash
# API health
curl http://YOUR_INSTANCE_IP:8000/health

# Individual service health
docker ps
```

## Database Migrations

### Automatic (during deployment)
Migrations run automatically during deployment via `deploy-gcp.sh`.

### Manual
```bash
# SSH into instance
gcloud compute ssh orin-production --zone=us-central1-a

# Run migrations
docker exec orin-server /app/server migrate
```

### Rollback
```bash
# Restore from backup
gsutil cp gs://YOUR_PROJECT-orin-backups/postgres/backup-YYYYMMDD.sql - | \
  docker exec -i orin-postgres psql -U postgres chunkr
```

## Troubleshooting

### Deployment Fails

1. **Check Cloud Build logs**
   ```bash
   gcloud builds list --limit=5
   gcloud builds log BUILD_ID
   ```

2. **Check instance status**
   ```bash
   gcloud compute instances describe orin-production --zone=us-central1-a
   ```

### Service Issues

1. **Check container logs**
   ```bash
   docker logs orin-server-1
   docker logs orin-task-1
   ```

2. **Restart services**
   ```bash
   docker-compose -f docker-compose.production.yaml restart
   ```

### Database Issues

1. **Check PostgreSQL status**
   ```bash
   docker exec orin-postgres pg_isready -U postgres
   ```

2. **Access PostgreSQL**
   ```bash
   docker exec -it orin-postgres psql -U postgres -d chunkr
   ```

## Cost Optimization

### Current Monthly Costs (Estimate)
- Compute Engine (n1-standard-8 + T4 GPU): $600-800
- Persistent Disk (SSD, 200GB): $40
- Cloud Storage (500GB): $10-15
- Network Egress: $50-200
- **Total: ~$700-1,055/month**

### Optimization Tips
1. Use preemptible instances for non-critical workloads (-80% cost)
2. Implement autoscaling for task workers
3. Use Cloud Storage lifecycle policies
4. Monitor and optimize GPU usage
5. Consider reserved instances for predictable workloads

## Security Best Practices

1. **Use Secret Manager** for all sensitive values
2. **Enable VPC** firewall rules
3. **Use IAM roles** instead of service account keys
4. **Enable Cloud Armor** for DDoS protection
5. **Regular security audits** with Cloud Security Command Center

## Future Improvements

### Phase 1: Managed Services
- Migrate to Cloud SQL for PostgreSQL
- Use Cloud Memorystore for Redis
- Native Cloud Storage integration

### Phase 2: Kubernetes
- Migrate to GKE for better orchestration
- Implement Horizontal Pod Autoscaling
- Multi-zone deployment for high availability

### Phase 3: Observability
- Cloud Monitoring integration
- Cloud Trace for distributed tracing
- Custom dashboards and alerting

## Support

For deployment issues:
- Email: support@orin.ai
- Documentation: See `gcp/README.md` for detailed GCP guide
- GitHub Issues: Report bugs and feature requests

## Changelog

### Version 1.0.0 (Current)
- Simplified from 7 compose files to 1
- Reduced replicas: 30→5 (task), 6→2 (segmentation), 3→2 (OCR)
- GCP-focused deployment
- Consolidated configuration files
- Simple Makefile workflow
- Removed self-hosting complexity


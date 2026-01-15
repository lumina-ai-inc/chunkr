# Orin GCP Deployment Guide

This directory contains all the configuration and scripts needed to deploy Orin to Google Cloud Platform.

## Prerequisites

1. **Google Cloud Project**
   - Create a GCP project with billing enabled
   - Enable the following APIs:
     - Compute Engine API
     - Container Registry API
     - Cloud Build API
     - Secret Manager API
     - Cloud Storage API

2. **Local Tools**
   - [gcloud CLI](https://cloud.google.com/sdk/docs/install)
   - [Terraform](https://www.terraform.io/downloads) (optional, for IaC)
   - Docker

3. **Configure gcloud**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud auth configure-docker
   ```

## Quick Deployment

### Option 1: Automated Deployment Script

```bash
# From project root
make deploy
```

Or manually:
```bash
./scripts/deploy-gcp.sh
```

This script will:
1. Build all container images using Cloud Build
2. Push images to Google Container Registry
3. Deploy to Compute Engine
4. Run database migrations
5. Perform health checks

### Option 2: Infrastructure as Code with Terraform

1. Initialize Terraform:
   ```bash
   cd gcp/terraform
   terraform init
   ```

2. Review the planned changes:
   ```bash
   terraform plan -var="project_id=YOUR_PROJECT_ID"
   ```

3. Apply the infrastructure:
   ```bash
   terraform apply -var="project_id=YOUR_PROJECT_ID"
   ```

4. Deploy the application:
   ```bash
   cd ../..
   make deploy
   ```

## Architecture

```
┌─────────────────────────────────────────┐
│         Cloud Load Balancer             │
│           (HTTPS Termination)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Compute Engine VM (n1-standard-8)  │
│         + NVIDIA Tesla T4 GPU           │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────────┐ │
│  │   Orin      │  │   Task Workers   │ │
│  │   Server    │  │   (x5)           │ │
│  │   (x2)      │  │                  │ │
│  └─────────────┘  └──────────────────┘ │
│                                          │
│  ┌─────────────┐  ┌──────────────────┐ │
│  │   OCR       │  │   Segmentation   │ │
│  │   Backend   │  │   Backend        │ │
│  │   (x2, GPU) │  │   (x2, GPU)      │ │
│  └─────────────┘  └──────────────────┘ │
│                                          │
│  ┌─────────────┐  ┌──────────────────┐ │
│  │  Postgres   │  │     Redis        │ │
│  └─────────────┘  └──────────────────┘ │
│                                          │
│  ┌─────────────┐  ┌──────────────────┐ │
│  │   MinIO     │  │     Qdrant       │ │
│  └─────────────┘  └──────────────────┘ │
│                                          │
│  ┌──────────────────────────────────┐  │
│  │         Keycloak                 │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Cloud Storage Buckets              │
│   - Documents (with versioning)         │
│   - Backups (with lifecycle policies)   │
└─────────────────────────────────────────┘
```

## Configuration

### Environment Variables

Set these in GCP Secret Manager:
- `ORIN_DB_PASSWORD` - PostgreSQL password
- `ORIN_API_KEYS` - LLM API keys
- `ORIN_STRIPE_KEY` - Stripe API key (optional)

### LLM Configuration

Upload your `models.yaml` to the VM:
```bash
gcloud compute scp models.yaml orin-production:~/models.yaml --zone=us-central1-a
```

## Monitoring & Operations

### View Logs
```bash
# SSH into the instance
gcloud compute ssh orin-production --zone=us-central1-a

# View container logs
docker logs -f orin-server
docker logs -f orin-task
```

### Scaling

#### Scale Task Workers
Edit `docker-compose.production.yaml` and update replicas:
```yaml
task:
  deploy:
    replicas: 10  # Increase as needed
```

Then redeploy:
```bash
make deploy
```

#### Scale Horizontally
Use Terraform to add more instances and configure load balancing.

### Backups

#### Database Backups
Automated daily backups to Cloud Storage:
```bash
# Manual backup
docker exec orin-postgres pg_dump -U postgres chunkr | \
  gsutil cp - gs://YOUR_PROJECT-orin-backups/postgres/backup-$(date +%Y%m%d).sql
```

#### Restore from Backup
```bash
gsutil cp gs://YOUR_PROJECT-orin-backups/postgres/backup-YYYYMMDD.sql - | \
  docker exec -i orin-postgres psql -U postgres chunkr
```

## Cost Estimation

Monthly costs (approximate):
- Compute Engine (n1-standard-8 + T4 GPU): $600-800
- Persistent Disk (SSD, 200GB): $40
- Cloud Storage (500GB): $10-15
- Egress (varies): $50-200
- **Total: ~$700-1,055/month**

### Cost Optimization Tips
1. Use preemptible instances for non-production workloads (-80% cost)
2. Implement autoscaling for task workers
3. Use Cloud Storage lifecycle policies for old documents
4. Consider Cloud SQL for easier management (slightly higher cost)

## Future Improvements

1. **Managed Services Migration**
   - Cloud SQL for PostgreSQL
   - Cloud Memorystore for Redis
   - Cloud Storage native integration

2. **Kubernetes Migration**
   - GKE for better orchestration
   - Horizontal Pod Autoscaling
   - Multi-zone deployment

3. **Observability**
   - Cloud Monitoring integration
   - Cloud Logging
   - Cloud Trace for distributed tracing

4. **Security**
   - Private Google Access
   - VPC Service Controls
   - Binary Authorization

## Troubleshooting

### Deployment Fails
```bash
# Check Cloud Build logs
gcloud builds list --limit=5

# Check instance logs
gcloud compute instances get-serial-port-output orin-production --zone=us-central1-a
```

### Health Check Failures
```bash
# SSH into instance
gcloud compute ssh orin-production --zone=us-central1-a

# Check container status
docker ps -a

# Check specific container logs
docker logs orin-server
```

### Database Connection Issues
```bash
# Verify PostgreSQL is running
docker exec orin-postgres pg_isready -U postgres

# Check database logs
docker logs orin-postgres
```

## Support

For deployment issues or questions:
- Email: support@orin.ai
- Documentation: https://docs.orin.ai
- GitHub Issues: https://github.com/buildorin/data-extract/issues


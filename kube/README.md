# Chunkr Kubernetes

## Prerequisites

- [Helm](https://helm.sh/docs/intro/install/)
- [Kubectl](https://kubernetes.io/docs/tasks/tools/)

### GPU Setup [Required]

#### For GKE Users
No additional setup required - GKE automatically handles NVIDIA drivers and device plugins for GPU nodes.

#### For Other Kubernetes Distributions
If you're not using GKE, follow these steps:

1. Install NVIDIA operator with time-slicing following the instructions at: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html#time-slicing-cluster-wide-config

```bash
# Add the NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
  && helm repo update

# Install the GPU Operator
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --version=v24.9.1

kubectl create -f time-slicing-config-all.yaml -n gpu-operator

kubectl patch clusterpolicy/cluster-policy \
  -n gpu-operator \
  --type merge \
  -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config-all", "default": "any"}}}}'
```

### Ingress Setup [Required]
Choose ONE of the following ingress methods:

#### Option 1: Cloudflare Tunnel [Recommended]
This option uses Cloudflare Tunnels for both ingress and SSL termination. This is recommended for simpler setup and better security.

Follow the setup instructions at: https://developers.cloudflare.com/cloudflare-one/tutorials/many-cfd-one-tunnel/

#### Option 2: NGINX Ingress Controller + Cloudflare SSL [In Development - Not Recommended]
This option uses NGINX for ingress and Cloudflare for SSL termination.

1. Install NGINX Ingress Controller:
```bash
# Check if NGINX ingress controller is already installed
kubectl get pods -A | grep nginx-ingress
# or
kubectl get ingressclass

# If not installed, you can install it using Helm:
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install nginx-ingress ingress-nginx/ingress-nginx
```

2. Set up Cloudflare SSL Certificate:
   - Log in to Cloudflare Dashboard and navigate to your domain
   - Go to SSL/TLS tab > Origin Server
   - Click "Create Certificate"
   - Choose "Let Cloudflare generate a private key and CSR"
   - Add your domain and subdomains (e.g., example.com, *.example.com)
   - Select RSA (2048) key type
   - Download the certificate files as `origin.crt` and `origin.key`
   - Create the TLS secret in Kubernetes:
   ```bash
   kubectl create secret tls tls-secret --cert=origin.crt --key=origin.key
   kubectl create secret tls tls-secret --cert=origin.crt --key=origin.key -n chunkr
   ```

## Installation

> **Note:**
> By default postgres, redis, and S3 use the filesystem. Optionally, you can use your own external providers. Click here to learn more about [external providers](#external-providers)


### 1. Setup Secrets

Create and configure your secrets:
```bash
# Create a secrets directory 
mkdir -p secrets/local

# Copy the example secrets
cp secrets/chunkr-secret.example.yaml secrets/local/chunkr-secret.yaml
```

Edit and apply your secrets:
```bash
# 1. Edit each secret file with your values
vim secrets/local/chunkr-secret.yaml  # or use your preferred editor

# 2. Create namespace and apply secrets
kubectl create namespace chunkr

# 3. Apply/update all secrets at once
kubectl apply -f secrets/local/ -n chunkr
```

### 2. Install with Helm

Choose one of the following installation methods:

**Basic Installation:**
```bash
helm install chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --create-namespace
```

**Custom Domain Installation with Cloudflare Tunnel:**
```bash
helm install chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --create-namespace \
  --set ingress.subdomains.root=false \
  --set "services.web.ingress.subdomain=chunkr" \
  --set "services.server.ingress.subdomain=chunkr-api" \
  --set "services.keycloak.ingress.subdomain=chunkr-auth" \
  --set "services.minio.ingress.subdomain=chunkr-s3" \
  --set ingress.type=cloudflare \
  --set cloudflared.enabled=true \
  --set cloudflared.config.tunnelName=YOUR_TUNNEL_NAME
```

**Installation with TLS (Cloudflare):**
```bash
helm install chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --create-namespace \
  --set ingress.tls.enabled=true \
  --set ingress.tls.secretName=tls-secret
```

## Update

To update the deployment, use one of the following methods:

**Basic Update:**
```bash
helm upgrade chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr
```

**Update with Configuration Changes:**
```bash
# Example: Update domain settings
helm upgrade chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set ingress.domain=new-domain.com \
  --set "services.web.ingress.subdomain=new-chunkr"
```

## Uninstall

```bash
helm uninstall chunkr --namespace chunkr
```

## External providers

### Storage Classes
By default, the storage class is set to "standard" which works for GCP. For other cloud providers, you'll need to specify the appropriate storage class:

- GCP: `standard`
- AWS: `gp2` or `gp3`
- Azure: `default` or `managed-premium`
- On-premise/Others: `default`

You can set the storage class during installation or upgrade:

```bash
# For GCP (default)
helm install chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set global.storageClass=standard

# For AWS
helm install chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set global.storageClass=gp2

# For Azure
helm install chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set global.storageClass=managed-premium
```

### S3 provider
By default, the S3 provider is set to MinIO. 
You must set the credentials for the external S3 provider in the chunkr-secret.yaml file.

```bash
# Update the chunkr-secret.yaml file with the credentials for the external S3 provider
AWS__ACCESS_KEY=
AWS__SECRET_KEY=
AWS__ENDPOINT=

# Disable MinIO
helm upgrade chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set services.minio.enabled=false
```

### Postgres

```bash
helm upgrade chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set services.postgres.enabled=false \
  --set "common.standardEnv[4].name=PG__URL" \
  --set "common.standardEnv[4].value=postgresql://user:password@your-external-postgres:5432/dbname"
```

### Redis

```bash
helm upgrade chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set services.redis.enabled=false \
  --set "common.standardEnv[6].name=REDIS__URL" \
  --set "common.standardEnv[6].value=redis://your-external-redis:6379"
```

## GPU Compatibility

The embeddings service supports different GPU architectures through specific Docker images. By default, it uses the Ampere 80 architecture (A100, A30, etc) with image `ghcr.io/huggingface/text-embeddings-inference:1.5`.

For the most up-to-date information about supported GPU architectures and their corresponding image tags, please refer to [Text Embeddings Inference Supported Models Documentation](https://huggingface.co/docs/text-embeddings-inference/supported_models#supported-hardware)

Example upgrade with GPU-specific image tag:

```bash
helm upgrade chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set services.embeddings.image.tag=turing-1.6  # Replace with your GPU-specific tag
```

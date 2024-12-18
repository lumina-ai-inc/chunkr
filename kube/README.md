# Chunkr Kubernetes

## Prerequisites

- [Helm](https://helm.sh/docs/intro/install/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)

### GPU Setup [Required]
Ensure the NVIDIA device plugin is installed:

```bash
# Check if NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia-device-plugin

# If no pods are found, install the plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Verify installation after a minute
kubectl get pods -n kube-system | grep nvidia-device-plugin
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

# Copy the example secret
cp secrets/chunkr-secret.example.yaml secrets/local/chunkr-secret.yaml
```

If using Cloudflare Tunnels:
```bash
# Additional secret needed for Cloudflare Tunnels
cp secrets/cloudflare-secret.example.yaml secrets/local/cloudflare-secret.yaml
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
  --namespace chunkr \
  --create-namespace
```

**Custom Domain Installation with Cloudflare Tunnel:**
```bash
helm install chunkr ./charts/chunkr \
  --namespace chunkr \
  --create-namespace \
  --set ingress.domain=example.com \
  --set ingress.subdomains.root=false \
  --set "services.web.ingress.subdomain=chunkr" \
  --set "services.chunkr.ingress.subdomain=chunkr-api" \
  --set "services.keycloak.ingress.subdomain=chunkr-auth" \
  --set "services.s3proxy.ingress.subdomain=chunkr-s3" \
  --set ingress.type=cloudflare \
  --set cloudflared.enabled=true
```

**Installation with TLS (Cloudflare):**
```bash
helm install chunkr ./charts/chunkr \
  --namespace chunkr \
  --create-namespace \
  --set ingress.tls.enabled=true \
  --set ingress.tls.secretName=tls-secret
```

## Update

To update the deployment, use one of the following methods:

**Basic Update:**
```bash
helm upgrade chunkr ./charts/chunkr --namespace chunkr
```

**Update with Configuration Changes:**
```bash
# Example: Update s3 provider to Azure
helm upgrade chunkr ./charts/chunkr \
  --namespace chunkr \
  --set global.s3provider=azure

# Example: Update domain settings
helm upgrade chunkr ./charts/chunkr \
  --namespace chunkr \
  --set ingress.domain=new-domain.com \
  --set "services.web.ingress.subdomain=new-chunkr"
```

## Uninstall

```bash
helm uninstall chunkr --namespace chunkr
```

## External providers

### S3 provider
You must set the credentials for the external S3 provider in the chunkr-secret.yaml file.

**AWS S3:**

```bash
helm upgrade chunkr ./charts/chunkr \
  --namespace chunkr \
  --set global.s3provider=aws
```

**GCP Storage (with Interoperability):**

```bash
helm upgrade chunkr ./charts/chunkr \
  --namespace chunkr \
  --set global.s3provider=gcp
```

**Azure Blob Storage:**

```bash
# Additional secret needed for Azure
cp secrets/azure-s3proxy-secret.example.yaml secrets/local/azure-s3proxy-secret.yaml

# Edit the secret file with your values
vim secrets/local/azure-s3proxy-secret.yaml

# Apply the secret
kubectl apply -f secrets/local/azure-s3proxy-secret.yaml -n chunkr

helm upgrade chunkr ./charts/chunkr \
  --namespace chunkr \
  --set global.s3provider=azure
```

### Postgres

```bash
helm upgrade chunkr ./charts/chunkr \
  --namespace chunkr \
  --set services.postgres.enabled=false \
  --set "common.standardEnv[4].name=PG__URL" \
  --set "common.standardEnv[4].value=postgresql://user:password@your-external-postgres:5432/dbname"
```

### Redis

```bash
helm upgrade chunkr ./charts/chunkr \
  --namespace chunkr \
  --set services.redis.enabled=false \
  --set "common.standardEnv[6].name=REDIS__URL" \
  --set "common.standardEnv[6].value=redis://your-external-redis:6379"
```

### GPU Compatibility

The embeddings service supports different GPU architectures through specific Docker images. By default, it uses the Turing architecture (T4, RTX 2000 series, etc) with image `ghcr.io/huggingface/text-embeddings-inference:turing-1.5` (experimental).

For the most up-to-date information about supported GPU architectures and their corresponding image tags, please refer to [Text Embeddings Inference Supported Models Documentation](https://huggingface.co/docs/text-embeddings-inference/supported_models#supported-hardware)

Example upgrade with GPU-specific image tag:

```bash
helm upgrade chunkr ./charts/chunkr \
  --namespace chunkr \
  --set services.embeddings.image.tag=1.5  # Replace with your GPU-specific tag
```
# Chunkr Kubernetes

## Prerequisites

- [Helm](https://helm.sh/docs/intro/install/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)

### GPU Setup [Required]
Ensure the NVIDIA device plugin is installed:

```bash
# Add the NVIDIA helm repository
helm repo add nvidia https://nvidia.github.io/k8s-device-plugin
helm repo update

# Install the plugin with the configuration
helm install nvidia-device-plugin nvidia/nvidia-device-plugin \
  --namespace kube-system \
  -f nvidia-values.yaml

helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin \
  --create-namespace \
  -f nvidia-values.yaml
```

You can change the number of replicas in the `nvidia-values.yaml` file.
```bash
# Increase to share with 8 pods
helm upgrade nvidia-device-plugin nvidia/nvidia-device-plugin \
  --namespace kube-system \
  --set "config.flags.timeSlicing.resources[0].replicas=8"

# Or decrease to share with 3 pods
helm upgrade nvidia-device-plugin nvidia/nvidia-device-plugin \
  --namespace kube-system \
  --set "config.flags.timeSlicing.resources[0].replicas=3"
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

### 1. Setup Secrets

Create and configure your secrets:
```bash
# Create a secrets directory 
mkdir -p secrets/local

# Copy the example secrets
cp secrets/chunkr-secret.example.yaml secrets/local/chunkr-secret.yaml
cp secrets/rrq-secret.example.yaml secrets/local/rrq-secret.yaml
cp secrets/keycloak-secret.example.yaml secrets/local/keycloak-secret.yaml
cp secrets/web-secret.example.yaml secrets/local/web-secret.yaml
```

If using Azure Storage:
```bash
# Additional secret needed for Azure
cp secrets/s3proxy-secret.example.yaml secrets/local/s3proxy-secret.yaml
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
helm install chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace
```

**Custom Domain Installation:**
```bash
helm install chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace \
  --set ingress.domain=example.com \
  --set ingress.subdomains.root=false \
  --set "services.web.ingress.subdomain=chunkr" \
  --set "services.chunkr.ingress.subdomain=chunkr-api" \
  --set "services.keycloak.ingress.subdomain=chunkr-auth" \
  --set "services.rrq.ingress.subdomain=chunkr-rrq-api" \
  --set "services.rrq-analytics.ingress.subdomain=chunkr-rrq" \
  --set "services.s3proxy.ingress.subdomain=chunkr-s3"
```

**Azure Installation:**
```bash
helm install chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace \
  --set global.provider=azure
```

**Cloudflare Tunnel Installation:**
```bash
helm install chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace \
  --set ingress.type=cloudflare \
  --set ingress.cloudflare.enabled=true
```

**Installation with TLS (Cloudflare):**
```bash
helm install chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace \
  --set ingress.tls.enabled=true \
  --set ingress.tls.secretName=tls-secret
```

## Update

To update the deployment, use one of the following methods:

**Basic Update:**
```bash
helm upgrade chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace
```

**Update with Configuration Changes:**
```bash
# Example: Update provider to Azure
helm upgrade chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace \
  --set global.provider=azure

# Example: Update domain settings
helm upgrade chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace \
  --set ingress.domain=new-domain.com \
  --set "services.web.ingress.subdomain=new-chunkr"
```

## Uninstall

```bash
helm uninstall chunkr --namespace chunkr
```

### GPU Compatibility

The embeddings service supports different GPU architectures through specific Docker images. By default, it uses the Turing architecture (T4, RTX 2000 series, etc) with image `ghcr.io/huggingface/text-embeddings-inference:turing-1.5` (experimental).

For the most up-to-date information about supported GPU architectures and their corresponding image tags, please refer to [Text Embeddings Inference Supported Models Documentation](https://huggingface.co/docs/text-embeddings-inference/supported_models#supported-hardware)

Example installation with GPU-specific image tag:

```bash
helm install chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace \
  --set services.embeddings.image.tag=1.5  # Replace with your GPU-specific tag
```
# Chunkr Kubernetes

## Prerequisites

- [Helm](https://helm.sh/docs/intro/install/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)

### GPU Setup
Ensure the NVIDIA device plugin is installed:

```bash
# Check if NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia-device-plugin

# If no pods are found, install the plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Verify installation after a minute
kubectl get pods -n kube-system | grep nvidia-device-plugin
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
  --set "services.chunkr.ingress.subdomain=chunkr-api" \
  --set "services.keycloak.ingress.subdomain=chunkr-auth" \
  --set "services.rrq.ingress.subdomain=chunkr-rrq-api" \
  --set "services.rrq-analytics.ingress.subdomain=chunkr-rrq"
```

**Azure Installation with custom domain:**
```bash
helm install chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace \
  --set global.provider=azure \
  --set "services.chunkr.ingress.subdomain=chunkr-api" \
  --set "services.keycloak.ingress.subdomain=chunkr-auth" \
  --set "services.rrq.ingress.subdomain=chunkr-rrq-api" \
  --set "services.rrq-analytics.ingress.subdomain=chunkr-rrq"
```

## Uninstall

```bash
helm uninstall chunkr --namespace chunkr
```

### GPU Compatibility

The embeddings service supports different GPU architectures through specific Docker images. Choose the appropriate image tag in your `values.yaml` based on your GPU:

| Architecture | Image Tag | Notes |
|--------------|-----------|--------|
| CPU | `cpu-1.5` | For non-GPU deployments |
| Volta | Not Supported | V100, Titan V, GTX 1000 series not supported |
| Turing | `turing-1.5` | For T4, RTX 2000 series (experimental) |
| Ampere 80 | `1.5` | For A100, A30 |
| Ampere 86 | `86-1.5` | For A10, A40 |
| Ada Lovelace | `89-1.5` | For RTX 4000 series |
| Hopper | `hopper-1.5` | For H100 (experimental) |

Example installation with GPU-specific image tag:
```bash
helm install chunkr ./chunkr-chart \
  --namespace chunkr \
  --create-namespace \
  --set services.embeddings.image.tag=1.5  # Replace with your GPU-specific tag
```



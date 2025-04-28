# Chunkr Kubernetes

## Table of Contents
- [Chunkr Kubernetes](#chunkr-kubernetes)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
    - [GPU Setup](#gpu-setup)
      - [For GKE Users](#for-gke-users)
      - [For Other Kubernetes Distributions](#for-other-kubernetes-distributions)
    - [Cloudflare Tunnel (Recommended)](#cloudflare-tunnel-recommended)
  - [Installation](#installation)
    - [1. Create Namespace](#1-create-namespace)
    - [2. Setup Secrets](#2-setup-secrets)
    - [3. Setup `models.yaml`](#3-setup-modelsyaml)
    - [4. Install with Helm](#4-install-with-helm)
  - [Update](#update)
  - [Uninstall](#uninstall)
  - [External providers](#external-providers)
    - [Storage Classes](#storage-classes)
    - [S3 provider](#s3-provider)
    - [Redis](#redis)
    - [Postgres](#postgres)

## Prerequisites

- [Helm](https://helm.sh/docs/intro/install/)
- [Kubectl](https://kubernetes.io/docs/tasks/tools/)

### GPU Setup

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

### Cloudflare Tunnel (Recommended)
This option uses Cloudflare Tunnels for both ingress and SSL termination. This is recommended for simpler setup and better security.

Follow the setup instructions at: https://developers.cloudflare.com/cloudflare-one/tutorials/many-cfd-one-tunnel/

## Installation

> **Note:**
> By default postgres, redis, and S3 use the filesystem. Optionally, you can use your own external providers. Click here to learn more about [external providers](#external-providers)

### 1. Create Namespace

```bash
kubectl create namespace chunkr
```

### 2. Setup Secrets

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
vim secrets/local/chunkr-secret.yaml  

# 2. Apply secrets
kubectl apply -f secrets/local/ -n chunkr
```

### 3. Setup `models.yaml`

Configure your models:
```bash
# Copy the example models.yaml
cp ../models.yaml.example secrets/local/models.yaml

# Edit the models.yaml file with your values
vim secrets/local/models.yaml

# Create the llm configmap
kubectl create configmap llm-models-configmap --from-file=models.yaml=./secrets/local/models.yaml -n chunkr
```

### 4. Install with Helm

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
  --set cloudflared.config.tunnelName=YOUR_TUNNEL_NAME \
  --set global.storageClass=standard
```

## Update

```bash
helm upgrade chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set ingress.subdomains.root=false \
  --set "services.web.ingress.subdomain=chunkr" \
  --set "services.server.ingress.subdomain=chunkr-api" \
  --set "services.keycloak.ingress.subdomain=chunkr-auth" \
  --set "services.minio.ingress.subdomain=chunkr-s3" \
  --set ingress.type=cloudflare \
  --set cloudflared.enabled=true \
  --set cloudflared.config.tunnelName=YOUR_TUNNEL_NAME \
  --set global.storageClass=standard
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

### Redis

Redis is managed in the cluster by default. You must set the credentials for the external Redis instance in the chunkr-secret.yaml file.

```bash
# Update the chunkr-secret.yaml file with the credentials for the external Redis instance
REDIS__URL=

# Disable Redis
helm upgrade chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set services.redis.enabled=false \
```


### Postgres

Postgres is disabled by default, a managed Postgres service is recommended. To enable it, run:

```bash
# Enable Postgres
helm upgrade chunkr ./charts/chunkr \
  -f ./charts/chunkr/values.yaml \
  -f ./charts/chunkr/infrastructure.yaml \
  --namespace chunkr \
  --set services.postgres.enabled=true \
  --set services.postgres.credentials.username={YOUR_USERNAME} \
  --set services.postgres.credentials.password={YOUR_PASSWORD}
```

```bash
# Update the chunkr-secret.yaml file with the credentials for the external Postgres instance
PG__URL=
```

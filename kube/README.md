# Chunkr Kubernetes

## Prerequisites

- [Helm](https://helm.sh/docs/intro/install/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)

### Setup Secrets

1. Create a local secrets directory next to your example files:
```bash
# Create a secrets directory 
mkdir -p secrets/local

# Copy the example secrets to your local directory
cp secrets/chunkr-secret.example.yaml secrets/local/chunkr-secret.yaml
cp secrets/rrq-secret.example.yaml secrets/local/rrq-secret.yaml
cp secrets/keycloak-secret.example.yaml secrets/local/keycloak-secret.yaml
cp secrets/web-secret.example.yaml secrets/local/web-secret.yaml
cp secrets/s3proxy-secret.example.yaml secrets/local/s3proxy-secret.yaml  # For Azure setup
```

2. Edit each file with your actual values
```bash
vim chunkr-secret.yaml  # or use your preferred editor
```

3. Apply or update secrets in your cluster:
```bash
# Create namespace if it doesn't exist
kubectl create namespace chunkr

# Apply/update all secrets at once
kubectl apply -f secrets/local/ -n chunkr
```

4. When you need to edit secrets later:
```bash
# Edit the secret file
vim secrets/local/chunkr-secret.yaml

# Reapply the specific secret
kubectl apply -f secrets/local/chunkr-secret.yaml -n chunkr
```
## Install

### Azure Setup
When deploying to Azure, you'll need to:
1. Configure the s3proxy-secret with your Azure storage credentials
2. Set the provider to "azure" during installation

```bash
# Basic install with default values
helm install chunkr ./chunkr-chart --namespace chunkr --create-namespace

# Install with custom configuration
helm install chunkr ./chunkr-chart --namespace chunkr --create-namespace \
  --set ingress.domain=example.com \
  --set "services.chunkr.ingress.subdomain=chunkr-api" \
  --set "services.keycloak.ingress.subdomain=chunkr-auth" \
  --set "services.rrq.ingress.subdomain=chunkr-rrq-api" \
  --set "services.rrq-analytics.ingress.subdomain=chunkr-rrq"

# Azure specific
helm install chunkr ./chunkr-chart --namespace chunkr --create-namespace \
  --set global.provider=azure \
  --set "services.chunkr.ingress.subdomain=chunkr-api" \
  --set "services.keycloak.ingress.subdomain=chunkr-auth" \
  --set "services.rrq.ingress.subdomain=chunkr-rrq-api" \
  --set "services.rrq-analytics.ingress.subdomain=chunkr-rrq"
```

### Modify Configuration or Upgrade Version

Use `helm upgrade` to either:
- Update configuration settings
- Upgrade to a newer version of the application

```bash
# Upgrade to a newer version
helm upgrade chunkr ./chunkr-chart --namespace chunkr

# Update configuration settings
helm upgrade chunkr ./chunkr-chart --namespace chunkr \
  --set global.provider=azure
```

## Uninstall

```bash
helm uninstall chunkr --namespace chunkr
```


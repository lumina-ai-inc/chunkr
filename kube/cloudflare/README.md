# Cloudflare Tunnel

## Cloudflare Tunnel Token

The Cloudflare Tunnel token is used to authenticate the Cloudflare Tunnel service.

The token is stored in the `cloudflared-secret` secret.

## Create the secret

```bash
cp cloudflared-secret.example.yaml cloudflared-secret.yaml
```

## Obtaining the Cloudflare Tunnel Token

1. Log in to your Cloudflare dashboard.
2. Navigate to the Zero Trust section.
3. Select Networks > Tunnels and follow the instructions to generate a token.

## Deploy to Kubernetes

### Create the namespace

```bash
kubectl create namespace cloudflare
```

### Apply the secret and cloudflared service

```bash
kubectl apply -f cloudflare-secret.yaml --namespace cloudflare
kubectl apply -f cloudflared.yaml --namespace cloudflare
```

### Verify the deployment

```bash
kubectl get pods --namespace cloudflare
```
    
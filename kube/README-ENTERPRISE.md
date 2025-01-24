### Enterprise Installation

# Create Docker registry secret for Docker Hub
```bash
kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=<your-dockerhub-username> \
  --docker-password=<your-dockerhub-password> \
  --namespace chunkr
```

For enterprise deployments with Azure integration:
```bash
helm install chunkr ./charts/chunkr \
  --namespace chunkr \
  --create-namespace \
  -f ./charts/chunkr/values-enterprise.yaml
``` 
# Self Deployment

## GCP (Google Cloud Platform)

### 1. Deploy Terraform

1. Log in to your Google Cloud account:
   ```bash
   gcloud auth login
   ```

2. Set your GCP project:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

3. Navigate to the GCP Terraform directory:
   ```bash
   cd terraform/gcp
   ```

4. Open the Terraform variables file:
   ```bash
   nano terraform.tfvars
   ```

5. Set the following variables in your `terraform.tfvars` file:

   #### Required Variables
   | Variable | Description |
   |----------|-------------|
   | `project` | Your GCP project ID |
   | `postgres_username` | Username for PostgreSQL |
   | `postgres_password` | Password for PostgreSQL |

   #### Optional Variables
   | Variable | Description | Default |
   |----------|-------------|---------|
   | `base_name` | Base name for resources | Provided |
   | `region` | GCP region for deployment | Provided |
   | `cluster_name` | Name of the Kubernetes cluster | Provided |
   | `chunkmydocs_db` | Name of the ChunkMyDocs database | Provided |
   | `keycloak_db` | Name of the Keycloak database | Provided |

   > **Note**: Optional variables have default values in the Terraform configuration. You can override these in your `terraform.tfvars` file to customize your deployment.

### 2. Setup Secrets

Copy the example secret files to your GCP configuration:

```bash
cp kube/secret/chunkmydocs-secret.example.yaml kube/gcp/chunkmydocs-secret.yaml
```
```bash
cp kube/secret/rrq-secret.example.yaml kube/gcp/rrq-secret.yaml
```
```bash
cp kube/secret/keycloak-secret.example.yaml kube/gcp/keycloak-secret.yaml
```

For each file, replace the placeholder values with your actual secret information. 

### 3. Deploy Kubernetes Resources

1. Configure kubectl to use your GCP cluster:
   ```bash
   gcloud container clusters get-credentials YOUR_CLUSTER_NAME --region YOUR_REGION
   ```

2. Create the `chunkmydocs` namespace:
   ```bash
   kubectl create namespace chunkmydocs
   ```

3. Apply the Kubernetes resources:
   ```bash
   kubectl apply -f kube/gcp/
   ```
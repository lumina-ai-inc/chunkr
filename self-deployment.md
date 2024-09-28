# Self Deployment

## GCP (Google Cloud Platform)

### 1. Install Google Cloud SDK

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)

2. Log in to your Google Cloud account:
   ```bash
   gcloud auth login
   ```

3. Set your GCP project:google-cloud-sdk 2
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

### 2. Install Terraform

1. Install [Terraform](https://developer.hashicorp.com/terraform/tutorials/gke/gke-install)

2. Navigate to the GCP Terraform directory:
   ```bash
   cd terraform/gcp
   ```

3. Open the Terraform variables file:
   ```bash
   nano terraform.tfvars
   ```

4. Set the following variables in your `terraform.tfvars` file:

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
   | `bucket_name` | Name of the GCS bucket | Provided |
   | `chunkmydocs_db` | Name of the ChunkMyDocs database | Provided |
   | `keycloak_db` | Name of the Keycloak database | Provided |

   > **Note**: Optional variables have default values in the Terraform configuration. You can override these in your `terraform.tfvars` file to customize your deployment.

5. Initialize Terraform:
   ```bash
   terraform init
   ```

6. Apply the Terraform configuration:
   ```bash
   terraform apply
   ```

7. Get raw output values:
   ```bash
   terraform output -json | jq -r 'to_entries[] | "echo \"\(.key): $(terraform output -raw \(.key))\"" ' | bash
   ```
   > **Note**: The output will contain sensitive information. Make sure to keep it secure.

8. Setup startup cron job on the VM to run the docker compose file:

   ```bash
   crontab -e
   ```

   Add the following line to the end of the file:

   ```bash
   @reboot sleep 60; cd /home/debian/ && sudo docker compose up -d
   ```

### 3. Setup Secrets

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
```bash
cp kube/secret/web-secret.example.yaml kube/gcp/web-secret.yaml
```
```bash
cp kube/secret/pdla-secret.example.yaml kube/gcp/pdla-secret.yaml
```


For each file, replace the placeholder values with your actual secret information. Use the values from the Terraform output.

### 4. Deploy Kubernetes Resources

1. Install kubectl following this [guide](https://kubernetes.io/docs/tasks/tools/)  

2. Configure kubectl to use your GCP cluster and region:
   ```bash
   gcloud container clusters get-credentials YOUR_CLUSTER_NAME --region YOUR_REGION
   ```
   > **Note**: This value is from the Terraform output `gke_connection_command`.

2. Create the `chunkmydocs` namespace:
   ```bash
   kubectl create namespace chunkmydocs
   ```

3. From the root of the repo, apply the Kubernetes resources:
   ```bash
   kubectl apply -R -f kube/gcp/
   ```

### 5. Finish the Deployment

1. Set up keycloak

2. Finish the secrets in the `chunkmydocs-secret.yaml` file:
   - `AUTH__KEYCLOAK_URL`
   - `AUTH__KEYCLOAK_REALM`
   - `EXTRACTION__BASE_URL`

3. Apply the Kubernetes resources:
   ```bash
   kubectl apply -R -f kube/gcp/
   ```
## 6. Cloudflare SSL certificate

1. Log in to Cloudflare Dashboard:
   - Navigate to your domain in the Cloudflare dashboard.

2. Access Origin Certificates:
   - Go to the SSL/TLS tab.
   - Click on Origin Server in the sub-menu.

3. Create a Certificate:
   - Click on Create Certificate.
   - Choose "Let Cloudflare generate a private key and CSR".
   - Under Hostnames, add your domain and any necessary subdomains (e.g., example.com, *.example.com).
   - Select Key Type: RSA (2048) is standard.
   - Set Certificate Validity as desired (default is 15 years).

4. Download Certificate and Key:
   - After creation, download the Origin Certificate and Private Key.
   - Save them as `origin.crt` and `origin.key` respectively.

5. Create a tls secret in kubernetes:
   ```bash
   kubectl create secret tls tls-secret --cert=kube/gcp/origin.crt --key=kube/gcp/origin.key -n chunkmydocs
   ```
6. Update `chunkmydocs-ingress.yaml`, `web-ingress.yaml`, and `keycloak-ingress.yaml` to use the tls secret:
   ```yaml
   tls:
   - hosts:
     - api.chunkr.ai
     secretName: tls-secret
   ```

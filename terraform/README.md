# Chunkr Terraform

## Prerequisites
- Install [Terraform](https://developer.hashicorp.com/terraform/tutorials/gke/gke-install)
- For GCP: Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- For Azure: Install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)

## GCP Deployment

1. Authenticate with GCP:
   ```bash
   gcloud auth application-default login
   ```

2. Navigate to the GCP Terraform directory:
   ```bash
   cd terraform/gcp
   ```

3. Create a `terraform.tfvars` file:
   ```bash
   nano terraform.tfvars
   ```

4. Set the following variables:

   #### Required Variables
   | Variable | Description |
   |----------|-------------|
   | `project` | Your GCP project ID |
   | `postgres_username` | Username for PostgreSQL |
   | `postgres_password` | Password for PostgreSQL |

   #### Optional Variables
   | Variable | Description | Default |
   |----------|-------------|---------|
   | `base_name` | Base name for resources | chunkmydocs |
   | `region` | GCP region | us-central1 |
   | `cluster_name` | GKE cluster name | chunkmydocs-cluster |
   | `bucket_name` | GCS bucket name | chunkmydocs-bucket |

## Azure Deployment

1. Authenticate with Azure:
   ```bash
   az login
   ```

2. Navigate to the Azure Terraform directory:
   ```bash
   cd terraform/azure
   ```

3. Create a `terraform.tfvars` file:
   ```bash
   nano terraform.tfvars
   ```

4. Set the following variables:

   #### Required Variables
   | Variable | Description |
   |----------|-------------|
   | `postgres_username` | Username for PostgreSQL |
   | `postgres_password` | Password for PostgreSQL |

   #### Optional Variables
   | Variable | Description | Default |
   |----------|-------------|---------|
   | `base_name` | Base name for resources | chunkr |
   | `location` | Azure region | eastus2 |

## Deployment Steps (Both Providers)

1. Initialize Terraform:
   ```bash
   terraform init
   ```

2. Review the planned changes:
   ```bash
   terraform plan
   ```

3. Apply the Terraform configuration:
   ```bash
   terraform apply
   ```

4. Get raw output values:
   ```bash
   terraform output -json | jq -r 'to_entries[] | "echo \"\(.key): $(terraform output -raw \(.key))\"" ' | bash
   ```
   > **Note**: The output will contain sensitive information. Make sure to keep it secure.

## Cleanup

To destroy all created resources:
```bash
terraform destroy
```

> **Warning**: This will permanently delete all resources created by Terraform.

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

### GCP

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

### Azure

4. Set the following variables:

   #### Required Variables
   | Variable | Description |
   |----------|-------------|
   | `base_name` | The base name for the resources |

   #### Optional Variables
   | Variable | Description | Default |
   |----------|-------------|---------|
   | `location` | Azure region | eastus2 |
   | `postgres_username` | Username for PostgreSQL | postgres |
   | `postgres_password` | Password for PostgreSQL | postgres |
   | `chunkr_db` | Name of the Chunkr database | chunkr |
   | `keycloak_db` | Name of the Keycloak database | keycloak |
   | `general_vm_count` | Number of general VMs | 1 |
   | `general_min_vm_count` | Minimum number of general VMs | 1 |
   | `general_max_vm_count` | Maximum number of general VMs | 1 |
   | `general_vm_size` | Size of general VMs | Standard_F8s_v2 |
   | `gpu_vm_count` | Number of GPU VMs | 1 |
   | `gpu_min_vm_count` | Minimum number of GPU VMs | 1 |
   | `gpu_max_vm_count` | Maximum number of GPU VMs | 1 |
   | `gpu_vm_size` | Size of GPU VMs | Standard_NC8as_T4_v3 |
   | `container_name` | Name of the storage container | chunkr |
   | `create_postgres` | Whether to create PostgreSQL resources | false |
   | `create_storage` | Whether to create Storage Account resources | true |

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

## S3 Backend Configuration (Optional)

Install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

1. Create a bucket and a dynamodb table

2. Copy the backend configuration example from terraform/{provider}:
   ```bash
   cp ../backends/backend.example.hcl ../backends/backend.hcl
   ```

3. Update the backend.hcl file with your S3 bucket details:
   - bucket: Your S3 bucket name
   - region: Your AWS region
   - dynamodb_table: Your DynamoDB table name
   - key: Update path if needed (defaults to provider/terraform.tfstate)

4. Initialize Terraform with your backend:
   ```bash
   terraform init -backend-config=../backends/backend.hcl
   ```


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
   | `base_name` | Base name for resources |
   | `project` | GCP project ID |

   #### Optional Variables
   | Variable | Description | Default |
   |----------|-------------|---------|
   | `region` | GCP region | us-central1 |
   | `postgres_username` | Username for PostgreSQL | postgres |
   | `postgres_password` | Password for PostgreSQL | postgres |

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

3. Apply the Terraform configuration:
```bash
terraform apply
```

4. Get raw output values:
```bash
terraform output -json | jq -r 'to_entries[] | "echo \"\(.key): $(terraform output -raw \(.key))\"" ' | bash
```
   > **Note**: The output will contain sensitive information. Make sure to keep it secure.

## Multiple Environments

### S3 Backend Configuration (Optional)

Install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

1. Create a bucket and a dynamodb table

2. Copy the backend configuration example from terraform/{provider}:
   
```bash
cp ../backend.example.hcl ./backend.hcl
```

3. Update the backend.hcl file with your S3 bucket details:
   - bucket: Your S3 bucket name
   - region: Your AWS region
   - dynamodb_table: Your DynamoDB table name
   - key: Update path if needed (defaults to provider/terraform.tfstate)

4. Initialize Terraform with your backend:
```bash
terraform init -backend-config=path/to/backend.hcl
```

**Other options:**

Migrate local state to backend:
```bash
terraform init -backend-config=path/to/backend.hcl -migrate-state
```

Use existing state from s3:
```bash
terraform init -backend-config=path/to/backend.hcl -reconfigure
```

### Environment Specific Configuration

1. Create a new terraform.tfvars file

2. Plan and apply the Terraform configuration:

```bash
terraform plan -var-file="path/to/terraform.tfvars" 
terraform apply -var-file="path/to/terraform.tfvars"
```

### Cleanup

To destroy resources from a specific backend:
```bash
# First reinitialize with the correct backend
terraform init -backend-config=path/to/backend.hcl -reconfigure

# Then destroy
terraform destroy -var-file="path/to/terraform.tfvars"
```

> **Warning**: This will permanently delete all resources created by Terraform.


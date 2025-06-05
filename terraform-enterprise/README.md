# Chunkr Terraform

## Prerequisites
- Install [Terraform](https://developer.hashicorp.com/terraform/tutorials/gke/gke-install)
- Install [Doppler CLI](https://docs.doppler.com/docs/install-cli) for secret management
- For GCP: Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- For Azure: Install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)

## GCP Deployment with Workspaces

The GCP deployment now uses Terraform workspaces for environment management and Doppler for secret management.

### Setup

1. **Authenticate with GCP:**
```bash
gcloud auth application-default login
```

2. **Set up Doppler:**
```bash
# Login to Doppler
doppler login

# Set up secrets for each environment
doppler secrets set postgres_username="your_username" --project chunkr-gcp --config prd
doppler secrets set postgres_password="your_password" --project chunkr-gcp --config prd

doppler secrets set postgres_username="dev_username" --project chunkr-gcp --config dev  
doppler secrets set postgres_password="dev_password" --project chunkr-gcp --config dev

doppler secrets set postgres_username="tbl_username" --project chunkr-gcp --config tbl
doppler secrets set postgres_password="tbl_password" --project chunkr-gcp --config tbl
```

3. **Navigate to the GCP directory:**
```bash
cd terraform-enterprise/gcp
```

### Deployment Commands

#### Using the convenience script (recommended):
```bash
# Make scripts executable (first time only)
chmod +x tf-workspace.sh

# Deploy to production
./tf-workspace.sh prod plan
./tf-workspace.sh prod apply

# Deploy to development  
./tf-workspace.sh dev plan
./tf-workspace.sh dev apply

# Deploy to turbolearn
./tf-workspace.sh turbolearn plan
./tf-workspace.sh turbolearn apply
```

#### Using Doppler run directly:
```bash
# Select workspace and run terraform
terraform workspace select prod
doppler run --project chunkr-gcp --config prd -- terraform plan
doppler run --project chunkr-gcp --config prd -- terraform apply

# For dev environment
terraform workspace select dev
doppler run --project chunkr-gcp --config dev -- terraform plan
doppler run --project chunkr-gcp --config dev -- terraform apply

# For turbolearn environment
terraform workspace select turbolearn
doppler run --project chunkr-gcp --config tbl -- terraform plan
doppler run --project chunkr-gcp --config tbl -- terraform apply
```

#### Manual token approach:
```bash
# Set up environment manually
terraform workspace select prod
export DOPPLER_TOKEN=$(doppler auth token --project chunkr-gcp --config prd)
terraform plan
terraform apply
```

### Environment Configuration

All environment-specific configurations are defined in the Terraform code itself:

| Environment | GCP Project | Region | Doppler Config |
|-------------|-------------|---------|----------------|
| `prod` | lumina-prod-424120 | us-west1 | prd |
| `dev` | lumina-prod-424120 | us-central1 | dev |
| `turbolearn` | turbolearn-ai | us-west1 | tbl |

### Required Doppler Secrets

For each environment, set these secrets in Doppler:

| Secret | Description |
|--------|-------------|
| `postgres_username` | Username for PostgreSQL |
| `postgres_password` | Password for PostgreSQL |

### Workspace Commands

```bash
# List available workspaces
terraform workspace list

# Show current workspace
terraform workspace show

# Switch to a different workspace
terraform workspace select <environment>

# Create a new workspace
terraform workspace new <environment>
```

### Migration from Old Setup

If you're migrating from the old backend file approach, use the migration script:

```bash
# Run the migration script
./migrate-to-workspaces.sh
```

See `WORKSPACE_MIGRATION.md` for detailed migration instructions.

## Azure Deployment

1. Authenticate with Azure:
```bash
az login
```

2. Navigate to the Azure Terraform directory:
```bash
cd terraform-enterprise/azure
```

3. Create a `terraform.tfvars` file:
```bash
nano terraform.tfvars
```

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

5. Apply the Terraform configuration:
```bash
terraform apply
```

## Getting Outputs

### GCP Workspaces
```bash
# Get outputs for current workspace
terraform output

# Get specific output
terraform output -raw cluster_name

# Get all outputs as JSON
terraform output -json
```

### Azure
```bash
terraform output -json | jq -r 'to_entries[] | "echo \"\(.key): $(terraform output -raw \(.key))\"" ' | bash
```
   > **Note**: The output will contain sensitive information. Make sure to keep it secure.

## State Management

### Backend Configuration

The GCP deployment uses a shared S3 backend with workspace-specific state files:

```
S3 Bucket: tf-backend-chunkr
├── workspaces/
│   ├── dev/
│   │   └── chunkr/gcp/terraform.tfstate
│   ├── prod/
│   │   └── chunkr/gcp/terraform.tfstate
│   └── turbolearn/
│       └── chunkr/gcp/terraform.tfstate
```

### Cleanup

To destroy resources from a specific environment:

```bash
# Using the convenience script
./tf-workspace.sh prod destroy

# Or manually
terraform workspace select prod
doppler run --project chunkr-gcp --config prd -- terraform destroy
```

> **Warning**: This will permanently delete all resources created by Terraform.

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy Infrastructure
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        type: choice
        options:
        - dev
        - prod
        - turbolearn

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v3
      
    - name: Install Doppler CLI
      uses: dopplerhq/cli-action@v3
      
    - name: Authenticate with GCP
      uses: google-github-actions/auth@v2
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}
        
    - name: Select Terraform Workspace
      run: |
        cd terraform-enterprise/gcp
        terraform init
        terraform workspace select ${{ github.event.inputs.environment }}
        
    - name: Terraform Plan
      run: |
        cd terraform-enterprise/gcp
        doppler run --project chunkr-gcp --config ${{ matrix.doppler_config }} -- terraform plan
      env:
        DOPPLER_TOKEN: ${{ secrets.DOPPLER_TOKEN }}
        
    - name: Terraform Apply
      run: |
        cd terraform-enterprise/gcp
        doppler run --project chunkr-gcp --config ${{ matrix.doppler_config }} -- terraform apply -auto-approve
      env:
        DOPPLER_TOKEN: ${{ secrets.DOPPLER_TOKEN }}
```

## Troubleshooting

### Common Issues

1. **Workspace not found**: Create it with `terraform workspace new <name>`
2. **Doppler authentication**: Run `doppler login` and ensure project/config exist
3. **State conflicts**: Ensure you're in the correct workspace with `terraform workspace show`

### Useful Commands

```bash
# Check current status
./tf-workspace.sh status

# Start an environment shell
./tf-workspace.sh prod shell

# Check Doppler connectivity
doppler auth token --project chunkr-gcp --config prd
```


# Terraform Workspace Migration Guide

## Table of Contents

- [Overview](#overview)
  - [Before (Current State)](#before-current-state)
  - [After (Target State)](#after-target-state)
- [Prerequisites](#prerequisites)
  - [1. Install Doppler CLI](#1-install-doppler-cli)
  - [2. Set up Doppler Authentication](#2-set-up-doppler-authentication)
  - [3. Configure Doppler Projects and Configs](#3-configure-doppler-projects-and-configs)
  - [4. Upload Secrets to Doppler](#4-upload-secrets-to-doppler)
- [Migration Process](#migration-process)
  - [Step 1: Run the Migration Script](#step-1-run-the-migration-script)
  - [Step 2: Test Each Workspace](#step-2-test-each-workspace)
  - [Step 3: Update CI/CD Pipelines](#step-3-update-cicd-pipelines)
- [New Workflow Commands](#new-workflow-commands)
  - [Working with Workspaces](#working-with-workspaces)
  - [Working with Doppler](#working-with-doppler)
  - [Environment-Specific Commands](#environment-specific-commands)
    - [Production](#production)
    - [Turbolearn](#turbolearn)
    - [Development](#development)
- [Configuration Structure](#configuration-structure)
  - [Workspace Configurations](#workspace-configurations)
  - [Secret Management](#secret-management)
- [State Management](#state-management)
  - [New State Structure](#new-state-structure)
  - [State Commands](#state-commands)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Rollback Plan](#rollback-plan)
- [Benefits of the New System](#benefits-of-the-new-system)
- [Cleanup](#cleanup)

This document guides you through migrating from separate backend configuration files to Terraform workspaces with Doppler integration.

## Overview

### Before (Current State)
- Separate `.hcl` backend files for each environment
- Separate `.tfvars` files for configuration
- Manual state management per environment

### After (Target State)
- Single Terraform configuration with workspaces
- Doppler for secret management
- Workspace-aware state management

## Prerequisites

### 1. Install Doppler CLI
```bash
# macOS
brew install dopplerhq/cli/doppler

# Other platforms: https://docs.doppler.com/docs/install-cli
```

### 2. Set up Doppler Authentication
```bash
doppler login
```

### 3. Configure Doppler Projects and Configs

You'll need to set up three configurations in Doppler:

```bash
# Create project (if it doesn't exist)
doppler projects create chunkr-infra

# Create configs for each environment
doppler configs create dev --project chunkr-infra
doppler configs create prd --project chunkr-infra  
doppler configs create tbl --project chunkr-infra
```

### 4. Upload Secrets to Doppler

For each environment, upload the required secrets:

```bash
# Production environment
doppler secrets set POSTGRES_USERNAME="chunkrprod" --project chunkr-infra --config prd
doppler secrets set POSTGRES_PASSWORD="{your_password}" --project chunkr-infra --config prd

# Turbolearn environment  
doppler secrets set POSTGRES_USERNAME="turbolearn" --project chunkr-infra --config tbl
doppler secrets set POSTGRES_PASSWORD="{your_password}" --project chunkr-infra --config tbl

# Dev environment (set your own credentials)
doppler secrets set POSTGRES_USERNAME="postgres" --project chunkr-infra --config dev
doppler secrets set POSTGRES_PASSWORD="postgres" --project chunkr-infra --config dev
```

## Migration Process

### Step 1: Run the Migration Script

```bash
# Make the script executable
chmod +x migrate-to-workspaces.sh

# Run the migration
./migrate-to-workspaces.sh
```

The script will:
1. Backup existing state files from each environment
2. Reconfigure the backend to use workspaces
3. Create workspaces for dev, prod, and turbolearn
4. Migrate state to the appropriate workspaces

### Step 2: Test Each Workspace

Test each environment to ensure the migration was successful:

```bash
# Test production
terraform workspace select prod
doppler run --name-transformer tf-var --project chunkr-infra --config prd -- terraform plan

# Test turbolearn
terraform workspace select turbolearn  
doppler run --name-transformer tf-var --project chunkr-infra --config tbl -- terraform plan

# Test dev
terraform workspace select dev
doppler run --name-transformer tf-var --project chunkr-infra  --config dev -- terraform plan
```

### Step 3: Update CI/CD Pipelines

Update your deployment pipelines to use workspaces:

```yaml
# Example GitHub Actions workflow
- name: Select Terraform Workspace
  run: terraform workspace select ${{ env.ENVIRONMENT }}

- name: Set Doppler Token
  run: echo "DOPPLER_TOKEN=${{ secrets.DOPPLER_TOKEN_PROD }}" >> $GITHUB_ENV
  
- name: Terraform Plan
  run: terraform plan
```

## New Workflow Commands

### Working with Workspaces

```bash
# List available workspaces
terraform workspace list

# Select a workspace
terraform workspace select <workspace_name>

# Create a new workspace
terraform workspace new <workspace_name>

# Show current workspace
terraform workspace show
```

### Working with Doppler

```bash
# Set environment token
export DOPPLER_TOKEN=$(doppler auth token --project chunkr-infra --config <env>)

# Run terraform commands
terraform plan
terraform apply
```

### Environment-Specific Commands

#### Production
```bash
terraform workspace select prod
doppler run --project chunkr-infra --config prd -- terraform plan
```

#### Turbolearn
```bash
terraform workspace select turbolearn
doppler run --project chunkr-infra --config tbl -- terraform plan
```

#### Development
```bash
terraform workspace select dev
doppler run --project chunkr-infra --config dev -- terraform plan
```

## Configuration Structure

### Workspace Configurations

All environment-specific configurations are now defined in `locals.workspace_configs`:

```hcl
locals {
  workspace_configs = {
    prod = {
      project              = "lumina-prod-424120"
      base_name            = "chunkr-prod"
      region               = "us-west1"
      # ... other config
    }
    dev = {
      project              = "lumina-prod-424120"
      base_name            = "chunkr-dev"
      region               = "us-central1"
      # ... other config
    }
    turbolearn = {
      project              = "turbolearn-ai"
      base_name            = "turbolearn-chunkr"
      region               = "us-west1"
      # ... other config
    }
  }
}
```

### Secret Management

Secrets are now managed through Doppler and accessed via:

```hcl
data "doppler_secrets" "this" {
  project = local.current_config.doppler_project
  config  = local.current_config.doppler_config
}

# Usage in resources
resource "google_sql_user" "users" {
  name     = data.doppler_secrets.this.data.POSTGRES_USERNAME
  password = data.doppler_secrets.this.data.POSTGRES_PASSWORD
  # ...
}
```

## State Management

### New State Structure

With workspaces, your state files are now organized as:

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

### State Commands

```bash
# View state in current workspace
terraform state list

# Move state between workspaces (if needed)
terraform workspace select source-workspace
terraform state mv <resource> <new-resource>
terraform workspace select target-workspace
terraform import <resource> <id>
```

## Troubleshooting

### Common Issues

1. **"No configuration found for workspace"**
   ```bash
   # Check if workspace exists in locals
   terraform console
   > local.workspace_configs
   ```

2. **Doppler authentication errors**
   ```bash
   # Re-authenticate
   doppler login
   # Check token
   doppler auth token --project chunkr-infra --config <env>
   ```

3. **State migration issues**
   ```bash
   # Check state backup
   ls -la state-backup/
   # Manually push state if needed
   terraform state push state-backup/terraform.<env>.tfstate
   ```

### Rollback Plan

If you need to rollback to the old system:

1. Keep the backup state files in `state-backup/`
2. Restore the old backend configuration files
3. Initialize with the old backend:
   ```bash
   terraform init -reconfigure -backend-config="backend.gcp.<env>.hcl"
   ```

## Benefits of the New System

1. **Simplified Management**: Single configuration file for all environments
2. **Better Security**: Secrets managed centrally in Doppler
3. **Workspace Isolation**: Clean separation between environments
4. **Easier CI/CD**: Standardized deployment patterns
5. **Version Control**: All configuration changes tracked in git

## Cleanup

Once you're confident the migration is successful:

1. Delete old backend files:
   ```bash
   rm backend.gcp.*.hcl
   ```

2. Delete old tfvars files:
   ```bash
   rm terraform.gcp.*.tfvars
   ```

3. Delete state backups (keep for a while as safety):
   ```bash
   # After several successful deployments
   rm -rf state-backup/
   ``` 
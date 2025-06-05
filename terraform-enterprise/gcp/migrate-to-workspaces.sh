#!/bin/bash

# Script to migrate from separate backend files to Terraform workspaces
# This script preserves existing state by copying it to workspace-specific locations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting migration to Terraform workspaces${NC}"

# Check if we're in the right directory
if [[ ! -f "main.tf" ]]; then
    echo -e "${RED}❌ Error: main.tf not found. Please run this script from the terraform-enterprise/gcp directory${NC}"
    exit 1
fi

# Check if Doppler CLI is installed
if ! command -v doppler &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Doppler CLI not found. Please install it from https://docs.doppler.com/docs/install-cli${NC}"
    echo "You'll need it to authenticate and access secrets."
fi

# Function to backup current state
backup_state() {
    local env=$1
    echo -e "${GREEN}📦 Backing up $env state...${NC}"
    
    # Create backup directory
    mkdir -p ./state-backup
    
    # Initialize with the old backend to pull state
    terraform init -reconfigure -backend-config="backend.gcp.$env.hcl"
    
    # Pull and backup state
    terraform state pull > "./state-backup/terraform.$env.tfstate"
    
    echo -e "${GREEN}✅ State backed up to ./state-backup/terraform.$env.tfstate${NC}"
}

# Function to migrate state to workspace
migrate_to_workspace() {
    local env=$1
    echo -e "${GREEN}🔄 Migrating $env to workspace...${NC}"
    
    # Initialize with new backend
    terraform init -reconfigure
    
    # Create workspace if it doesn't exist
    terraform workspace new $env 2>/dev/null || terraform workspace select $env
    
    # Push the backed up state to the new workspace
    terraform state push "./state-backup/terraform.$env.tfstate"
    
    echo -e "${GREEN}✅ Successfully migrated $env to workspace${NC}"
}

# Main migration process
main() {
    echo -e "${YELLOW}📋 This script will:${NC}"
    echo "1. Backup existing state files"
    echo "2. Reconfigure backend to use workspaces"
    echo "3. Create workspaces for each environment"
    echo "4. Migrate state to appropriate workspaces"
    echo ""
    
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Migration cancelled.${NC}"
        exit 0
    fi
    
    # List of environments to migrate
    environments=("dev" "prod" "turbolearn")
    
    # Step 1: Backup all existing states
    echo -e "${GREEN}📦 Step 1: Backing up existing states${NC}"
    for env in "${environments[@]}"; do
        if [[ -f "backend.gcp.$env.hcl" ]]; then
            backup_state $env
        else
            echo -e "${YELLOW}⚠️  Warning: backend.gcp.$env.hcl not found, skipping $env${NC}"
        fi
    done
    
    # Step 2: Migrate to workspaces
    echo -e "${GREEN}🔄 Step 2: Migrating to workspaces${NC}"
    for env in "${environments[@]}"; do
        if [[ -f "./state-backup/terraform.$env.tfstate" ]]; then
            migrate_to_workspace $env
        else
            echo -e "${YELLOW}⚠️  Warning: No backup found for $env, skipping${NC}"
        fi
    done
    
    echo -e "${GREEN}🎉 Migration completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}📝 Next steps:${NC}"
    echo "1. Set up your Doppler configurations for each environment"
    echo "2. Test each workspace with 'terraform workspace select <env>' and 'terraform plan'"
    echo "3. Delete old backend files once you're confident everything works"
    echo "4. Update your CI/CD pipelines to use workspaces"
}

# Run main function
main 
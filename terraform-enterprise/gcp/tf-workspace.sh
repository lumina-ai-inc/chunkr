#!/bin/bash

# Convenience script for Terraform workspace operations with Doppler integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo -e "${BLUE}🚀 Terraform Workspace Helper${NC}"
    echo ""
    echo "Usage: $0 <environment> <command> [args...]"
    echo ""
    echo "Environments: dev, staging, prod, turbolearn"
    echo "Commands:"
    echo "  plan                 - Run terraform plan"
    echo "  apply [args]         - Run terraform apply"
    echo "  destroy [args]       - Run terraform destroy"
    echo "  init                 - Initialize terraform"
    echo "  shell                - Start shell with environment variables set"
    echo "  status               - Show current workspace and configuration"
    echo ""
    echo "Location Override Options (for DR testing):"
    echo "  --override-region <region>        - Override region"
    echo "  --override-zone <zone_suffix>     - Override zone suffix" 
    echo "  --override-project <project>      - Override project"
    echo ""
    echo "Examples:"
    echo "  $0 prod plan"
    echo "  $0 dev apply"
    echo "  $0 staging destroy"
    echo "  $0 prod shell"
    echo "  $0 staging plan --override-region us-west2 --override-zone c"
}

# Function to parse override arguments
parse_overrides() {
    OVERRIDE_ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --override-region)
                OVERRIDE_ARGS="$OVERRIDE_ARGS -var override_region=$2"
                shift 2
                ;;
            --override-zone)
                OVERRIDE_ARGS="$OVERRIDE_ARGS -var override_zone_suffix=$2" 
                shift 2
                ;;
            --override-project)
                OVERRIDE_ARGS="$OVERRIDE_ARGS -var override_project=$2"
                shift 2
                ;;
            *)
                # Return remaining args
                echo "$@"
                return
                ;;
        esac
    done
}

# Function to set up environment
setup_env() {
    local env=$1
    
    # Doppler config mapping
    case $env in
        prod)
            DOPPLER_CONFIG="prd"
            ;;
        dev)
            DOPPLER_CONFIG="dev"
            ;;
        staging)
            DOPPLER_CONFIG="dev"  # Use dev config for staging
            ;;
        turbolearn)
            DOPPLER_CONFIG="tbl"
            ;;
        *)
            echo -e "${RED}❌ Error: Unknown environment '$env'${NC}"
            echo "Valid environments: dev, staging, prod, turbolearn"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}🔧 Setting up environment: $env${NC}"
    
    # Select workspace
    echo -e "${BLUE}📂 Selecting workspace: $env${NC}"
    terraform workspace select $env 2>/dev/null || {
        echo -e "${YELLOW}⚠️  Workspace '$env' doesn't exist. Creating it...${NC}"
        terraform workspace new $env
    }
    
    echo -e "${GREEN}✅ Environment setup complete${NC}"
}

# Function to run terraform command with doppler
run_terraform() {
    local env=$1
    local cmd=$2
    shift 2
    
    # Parse override arguments
    local remaining_args=$(parse_overrides "$@")
    
    setup_env $env
    
    echo -e "${BLUE}🏃 Running: doppler run --name-transformer tf-var --project chunkr-infra --config $DOPPLER_CONFIG -- terraform $cmd $OVERRIDE_ARGS $remaining_args${NC}"
    echo ""
    
    case $cmd in
        plan)
            doppler run --name-transformer tf-var --project chunkr-infra --config $DOPPLER_CONFIG -- terraform plan $OVERRIDE_ARGS $remaining_args
            ;;
        apply)
            doppler run --name-transformer tf-var --project chunkr-infra --config $DOPPLER_CONFIG -- terraform apply $OVERRIDE_ARGS $remaining_args
            ;;
        destroy)
            doppler run --name-transformer tf-var --project chunkr-infra --config $DOPPLER_CONFIG -- terraform destroy $OVERRIDE_ARGS $remaining_args
            ;;
        init)
            terraform init $remaining_args
            ;;
        *)
            echo -e "${RED}❌ Error: Unknown terraform command '$cmd'${NC}"
            exit 1
            ;;
    esac
}

# Function to start shell with environment
start_shell() {
    local env=$1
    
    setup_env $env
    
    echo -e "${GREEN}🐚 Starting shell with Doppler environment${NC}"
    echo "Environment: $env"
    echo "Workspace: $(terraform workspace show)"
    echo "Doppler Project: chunkr-infra"
    echo "Doppler Config: $DOPPLER_CONFIG"
    echo ""
    echo "You can now run terraform commands directly with doppler:"
    echo "  doppler run --name-transformer tf-var --project chunkr-infra --config $DOPPLER_CONFIG -- terraform plan"
    echo "  doppler run --name-transformer tf-var --project chunkr-infra --config $DOPPLER_CONFIG -- terraform apply"
    echo ""
    echo -e "${YELLOW}Type 'exit' to return to the normal shell${NC}"
    echo ""
    
    # Start new shell with environment
    PROMPT_COMMAND="PS1='[tf:$env] \u@\h:\w\$ '" bash --norc
}

# Function to show status
show_status() {
    local env=$1
    
    echo -e "${BLUE}📊 Terraform Workspace Status${NC}"
    echo ""
    
    # Current workspace
    echo -e "${GREEN}Current Workspace:${NC} $(terraform workspace show 2>/dev/null || echo 'Not initialized')"
    
    # Available workspaces
    echo -e "${GREEN}Available Workspaces:${NC}"
    terraform workspace list 2>/dev/null || echo "  Not initialized"
    
    # If environment specified, show doppler info
    if [[ -n "$env" ]]; then
        case $env in
            prod) DOPPLER_CONFIG="prd" ;;
            dev) DOPPLER_CONFIG="dev" ;;
            staging) DOPPLER_CONFIG="dev" ;;
            turbolearn) DOPPLER_CONFIG="tbl" ;;
            *) DOPPLER_CONFIG="unknown" ;;
        esac
        
        echo ""
        echo -e "${GREEN}Environment Configuration:${NC}"
        echo "  Environment: $env"
        echo "  Doppler Project: chunkr-infra"
        echo "  Doppler Config: $DOPPLER_CONFIG"
        
        # Test doppler connectivity
        if doppler run --name-transformer tf-var --project chunkr-infra --config $DOPPLER_CONFIG -- echo "test" >/dev/null 2>&1; then
            echo -e "  Doppler Status: ${GREEN}✅ Connected${NC}"
        else
            echo -e "  Doppler Status: ${RED}❌ Not authenticated${NC}"
        fi
    fi
    
    echo ""
}

# Main script logic
if [[ $# -lt 1 ]]; then
    show_usage
    exit 1
fi

# Special case for status command without environment
if [[ $1 == "status" && $# -eq 1 ]]; then
    show_status
    exit 0
fi

if [[ $# -lt 2 ]]; then
    show_usage
    exit 1
fi

ENV=$1
CMD=$2
shift 2
ARGS="$@"

case $CMD in
    plan|apply|destroy|init)
        run_terraform $ENV $CMD $ARGS
        ;;
    shell)
        start_shell $ENV
        ;;
    status)
        show_status $ENV
        ;;
    *)
        echo -e "${RED}❌ Error: Unknown command '$CMD'${NC}"
        show_usage
        exit 1
        ;;
esac 
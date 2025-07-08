#!/bin/bash

# Simple Terraform wrapper for Teleport infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo -e "${BLUE}🚀 Teleport Infrastructure Helper${NC}"
    echo ""
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  init                 - Initialize terraform"
    echo "  plan                 - Run terraform plan"
    echo "  apply [args]         - Run terraform apply"
    echo "  destroy [args]       - Run terraform destroy"
    echo "  output               - Show terraform outputs"
    echo "  status               - Show current status"
    echo ""
    echo "Location Override Options:"
    echo "  --override-region <region>        - Override region (default: us-central1)"
    echo "  --override-zone <zone_suffix>     - Override zone suffix (default: b)"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 plan"
    echo "  $0 apply"
    echo "  $0 destroy"
    echo "  $0 plan --override-region us-west1 --override-zone c"
    echo "  $0 output"
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
            *)
                # Return remaining args
                echo "$@"
                return
                ;;
        esac
    done
}

# Function to run terraform command
run_terraform() {
    local cmd=$1
    shift
    
    # Parse override arguments
    local remaining_args=$(parse_overrides "$@")
    
    echo -e "${BLUE}🏃 Running: terraform $cmd $OVERRIDE_ARGS $remaining_args${NC}"
    echo ""
    
    case $cmd in
        init)
            terraform init $remaining_args
            ;;
        plan)
            terraform plan $OVERRIDE_ARGS $remaining_args
            ;;
        apply)
            terraform apply $OVERRIDE_ARGS $remaining_args
            ;;
        destroy)
            terraform destroy $OVERRIDE_ARGS $remaining_args
            ;;
        output)
            terraform output $remaining_args
            ;;
        *)
            echo -e "${RED}❌ Error: Unknown terraform command '$cmd'${NC}"
            exit 1
            ;;
    esac
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Teleport Infrastructure Status${NC}"
    echo ""
    
    # Check if terraform is initialized
    if [[ -d ".terraform" ]]; then
        echo -e "${GREEN}✅ Terraform initialized${NC}"
    else
        echo -e "${RED}❌ Terraform not initialized${NC}"
        echo "Run: $0 init"
        echo ""
        return
    fi
    
    # Show current state
    echo -e "${GREEN}Current Configuration:${NC}"
    echo "  Project: lumina-prod-424120"
    echo "  Region: us-central1 (default)"
    echo "  Zone: us-central1-b (default)"
    echo "  Machine Type: n2-standard-2"
    echo "  Disk Size: 50GB"
    echo ""
    
    # Check if resources exist
    if terraform show -json > /dev/null 2>&1; then
        echo -e "${GREEN}Infrastructure Status:${NC}"
        
        # Check for VM
        if terraform show -json | jq -r '.values.root_module.resources[] | select(.type=="google_compute_instance") | .values.name' 2>/dev/null | grep -q "teleport-server"; then
            echo -e "  VM: ${GREEN}✅ Running${NC}"
        else
            echo -e "  VM: ${YELLOW}⚠️  Not found${NC}"
        fi
        
        # Check for IP
        if terraform show -json | jq -r '.values.root_module.resources[] | select(.type=="google_compute_address") | .values.address' 2>/dev/null | grep -q "."; then
            IP=$(terraform show -json | jq -r '.values.root_module.resources[] | select(.type=="google_compute_address") | .values.address' 2>/dev/null)
            echo -e "  IP Address: ${GREEN}✅ $IP${NC}"
        else
            echo -e "  IP Address: ${YELLOW}⚠️  Not allocated${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No infrastructure deployed${NC}"
    fi
    
    echo ""
}

# Function to show post-deployment info
show_post_deployment() {
    echo ""
    echo -e "${GREEN}🎉 Teleport Infrastructure Deployed Successfully!${NC}"
    echo ""
    
    # Get outputs
    TELEPORT_IP=$(terraform output -raw teleport_server_ip 2>/dev/null || echo "Not available")
    TELEPORT_URL=$(terraform output -raw teleport_web_url 2>/dev/null || echo "Not available")
    SSH_COMMAND=$(terraform output -raw ssh_command 2>/dev/null || echo "Not available")
    
    echo -e "${BLUE}📋 Access Information:${NC}"
    echo "  🌐 Web Interface: $TELEPORT_URL"
    echo "  🖥️  SSH Command: $SSH_COMMAND"
    echo "  📍 Server IP: $TELEPORT_IP"
    echo ""
    
    echo -e "${BLUE}🚀 Next Steps:${NC}"
    echo "1. SSH into the VM:"
    echo "   $SSH_COMMAND"
    echo ""
    echo "2. Create admin user:"
    echo "   ./create-admin-user.sh your-username"
    echo ""
    echo "3. Access web interface:"
    echo "   $TELEPORT_URL"
    echo ""
    echo "4. Create tokens for agents:"
    echo "   ./create-token.sh"
    echo ""
    echo "5. Read the setup guide:"
    echo "   cat README.md"
    echo ""
}

# Main script logic
if [[ $# -lt 1 ]]; then
    show_usage
    exit 1
fi

CMD=$1
shift
ARGS="$@"

case $CMD in
    init|plan|destroy|output)
        run_terraform $CMD $ARGS
        ;;
    apply)
        run_terraform $CMD $ARGS
        # Show post-deployment info if apply was successful
        if [[ $? -eq 0 ]]; then
            show_post_deployment
        fi
        ;;
    status)
        show_status
        ;;
    *)
        echo -e "${RED}❌ Error: Unknown command '$CMD'${NC}"
        show_usage
        exit 1
        ;;
esac 
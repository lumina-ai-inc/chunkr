#!/bin/bash

# Terraform wrapper for Teleport Agent infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo -e "${BLUE}🤖 Teleport Agent Infrastructure Helper${NC}"
    echo ""
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  init                           - Initialize terraform"
    echo "  plan                           - Run terraform plan"
    echo "  apply [args]                   - Run terraform apply"
    echo "  destroy [args]                 - Run terraform destroy"
    echo "  output                         - Show terraform outputs"
    echo "  status                         - Show current status"
    echo "  check-agents                   - Check agent connectivity"
    echo ""
    echo "Required Variables:"
    echo "  --teleport-server-ip <ip>      - IP of the Teleport server"
    echo "  --teleport-token <token>       - Token for agents to connect"
    echo ""
    echo "Optional Variables:"
    echo "  --agent-count <number>         - Number of agent VMs (default: 3)"
    echo "  --machine-type <type>          - VM machine type (default: n2-standard-2)"
    echo "  --disk-size <gb>               - Disk size in GB (default: 30)"
    echo "  --override-region <region>     - Override region"
    echo "  --override-zone <zone_suffix>  - Override zone suffix"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 plan --teleport-server-ip 1.2.3.4 --teleport-token abc123"
    echo "  $0 apply --teleport-server-ip 1.2.3.4 --teleport-token abc123 --agent-count 5"
    echo "  $0 check-agents"
    echo "  $0 destroy"
}

# Global variables for parsed arguments
TELEPORT_SERVER_IP=""
TELEPORT_TOKEN=""
AGENT_COUNT=""
MACHINE_TYPE=""
DISK_SIZE=""
OVERRIDE_REGION=""
OVERRIDE_ZONE=""
TF_ARGS=""

# Function to parse arguments
parse_args() {
    # Reset global variables
    TELEPORT_SERVER_IP=""
    TELEPORT_TOKEN=""
    AGENT_COUNT=""
    MACHINE_TYPE=""
    DISK_SIZE=""
    OVERRIDE_REGION=""
    OVERRIDE_ZONE=""
    TF_ARGS=""
    
    local remaining_args=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --teleport-server-ip)
                TELEPORT_SERVER_IP="$2"
                TF_ARGS="$TF_ARGS -var teleport_server_ip=$2"
                shift 2
                ;;
            --teleport-token)
                TELEPORT_TOKEN="$2"
                TF_ARGS="$TF_ARGS -var teleport_token=$2"
                shift 2
                ;;
            --agent-count)
                AGENT_COUNT="$2"
                TF_ARGS="$TF_ARGS -var agent_count=$2"
                shift 2
                ;;
            --machine-type)
                MACHINE_TYPE="$2"
                TF_ARGS="$TF_ARGS -var agent_machine_type=$2"
                shift 2
                ;;
            --disk-size)
                DISK_SIZE="$2"
                TF_ARGS="$TF_ARGS -var agent_disk_size_gb=$2"
                shift 2
                ;;
            --override-region)
                OVERRIDE_REGION="$2"
                TF_ARGS="$TF_ARGS -var override_region=$2"
                shift 2
                ;;
            --override-zone)
                OVERRIDE_ZONE="$2"
                TF_ARGS="$TF_ARGS -var override_zone_suffix=$2"
                shift 2
                ;;
            *)
                # Collect remaining args
                remaining_args="$remaining_args $1"
                shift
                ;;
        esac
    done
    
    # Return remaining args
    echo "$remaining_args"
}

# Function to validate required variables
validate_required() {
    local cmd=$1
    
    if [[ "$cmd" == "plan" || "$cmd" == "apply" ]]; then
        if [[ -z "$TELEPORT_SERVER_IP" ]]; then
            echo -e "${RED}❌ Error: --teleport-server-ip is required${NC}"
            echo "Get it from the client deployment: cd ../client && ./tf-teleport.sh output"
            exit 1
        fi
        
        if [[ -z "$TELEPORT_TOKEN" ]]; then
            echo -e "${RED}❌ Error: --teleport-token is required${NC}"
            echo "Get it from the Teleport server by SSHing in and running: ./create-token.sh"
            exit 1
        fi
    fi
}

# Function to run terraform command
run_terraform() {
    local cmd=$1
    shift
    
    # Parse arguments directly to set global variables
    parse_args "$@" >/dev/null
    
    validate_required $cmd
    
    echo -e "${BLUE}🏃 Running: terraform $cmd $TF_ARGS${NC}"
    echo ""
    
    case $cmd in
        init)
            terraform init
            ;;
        plan)
            terraform plan $TF_ARGS
            ;;
        apply)
            terraform apply $TF_ARGS
            ;;
        destroy)
            terraform destroy $TF_ARGS
            ;;
        output)
            terraform output
            ;;
        *)
            echo -e "${RED}❌ Error: Unknown terraform command '$cmd'${NC}"
            exit 1
            ;;
    esac
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Teleport Agent Infrastructure Status${NC}"
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
    echo "  Machine Type: n2-standard-2 (default)"
    echo "  Disk Size: 30GB (default)"
    echo "  Agent Count: 3 (default)"
    echo ""
    
    # Check if resources exist
    if terraform show -json > /dev/null 2>&1; then
        echo -e "${GREEN}Infrastructure Status:${NC}"
        
        # Check for VMs
        AGENT_COUNT=$(terraform show -json 2>/dev/null | jq -r '.values.root_module.resources[] | select(.type=="google_compute_instance") | .values.name' | wc -l)
        if [[ $AGENT_COUNT -gt 0 ]]; then
            echo -e "  Agent VMs: ${GREEN}✅ $AGENT_COUNT running${NC}"
            
            # Show VM details
            terraform show -json 2>/dev/null | jq -r '.values.root_module.resources[] | select(.type=="google_compute_instance") | "\(.values.name): \(.values.network_interface[0].network_ip)"' | while read line; do
                echo "    - $line"
            done
        else
            echo -e "  Agent VMs: ${YELLOW}⚠️  None found${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No infrastructure deployed${NC}"
    fi
    
    echo ""
}

# Function to check agent connectivity
check_agents() {
    echo -e "${BLUE}🔍 Checking Agent Connectivity${NC}"
    echo ""
    
    if ! terraform show -json > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  No infrastructure deployed${NC}"
        return
    fi
    
    # Get agent IAP SSH commands
    SSH_COMMANDS=$(terraform output -json agent_iap_ssh_commands 2>/dev/null | jq -r '.[]' 2>/dev/null || echo "")
    
    if [[ -z "$SSH_COMMANDS" ]]; then
        echo -e "${YELLOW}⚠️  No agent VMs found${NC}"
        return
    fi
    
    echo -e "${GREEN}Checking agent status...${NC}"
    echo ""
    
    while IFS= read -r ssh_cmd; do
        if [[ -n "$ssh_cmd" ]]; then
            VM_NAME=$(echo "$ssh_cmd" | grep -o 'teleport-agent-[0-9]*')
            echo -e "${BLUE}Checking $VM_NAME:${NC}"
            
            # Check if agent is running
            $ssh_cmd --command 'docker ps | grep teleport-agent' 2>/dev/null && echo -e "  ${GREEN}✅ Teleport agent running${NC}" || echo -e "  ${RED}❌ Teleport agent not running${NC}"
            
            # Check services
            $ssh_cmd --command 'curl -s localhost/health' >/dev/null 2>&1 && echo -e "  ${GREEN}✅ Web service healthy${NC}" || echo -e "  ${RED}❌ Web service unhealthy${NC}"
            
            $ssh_cmd --command 'curl -s localhost:3000/health' >/dev/null 2>&1 && echo -e "  ${GREEN}✅ Node.js service healthy${NC}" || echo -e "  ${RED}❌ Node.js service unhealthy${NC}"
            
            echo ""
        fi
    done <<< "$SSH_COMMANDS"
    
    echo -e "${BLUE}💡 Tip: Check these VMs in your Teleport web interface under 'Servers'${NC}"
}

# Function to show post-deployment info
show_post_deployment() {
    echo ""
    echo -e "${GREEN}🎉 Teleport Agent Infrastructure Deployed Successfully!${NC}"
    echo ""
    
    # Get outputs
    AGENT_COUNT=$(terraform output -raw agent_count 2>/dev/null || echo "Unknown")
    AGENT_NAMES=$(terraform output -json agent_vm_names 2>/dev/null | jq -r '.[]' 2>/dev/null | tr '\n' ' ' || echo "Unknown")
    TELEPORT_CONNECTION=$(terraform output -raw teleport_server_connection 2>/dev/null || echo "Unknown")
    
    echo -e "${BLUE}📋 Deployment Summary:${NC}"
    echo "  🤖 Agent VMs: $AGENT_COUNT"
    echo "  📡 Teleport Server: $TELEPORT_CONNECTION"
    echo "  🏷️  VM Names: $AGENT_NAMES"
    echo ""
    
    echo -e "${BLUE}🚀 Next Steps:${NC}"
    echo "1. Check agent connectivity:"
    echo "   $0 check-agents"
    echo ""
    echo "2. Access agents via Teleport:"
    echo "   - Open Teleport web interface"
    echo "   - Go to 'Servers' section"
    echo "   - Look for your agent VMs"
    echo "   - Click 'Connect' for terminal access"
    echo ""
    echo "3. Test web services on agents:"
    echo "   - Access via port forwarding through Teleport"
    echo "   - Or check internal IPs from Teleport terminal"
    echo ""
    echo "4. View detailed outputs:"
    echo "   $0 output"
    echo ""
}

# Function to suggest getting required values
suggest_values() {
    echo -e "${BLUE}💡 Getting Required Values:${NC}"
    echo ""
    echo "1. Get Teleport server IP:"
    echo "   cd ../client && ./tf-teleport.sh output | grep teleport_server_ip"
    echo ""
    echo "2. Get Teleport token:"
    echo "   cd ../client"
    echo "   # SSH into Teleport server using output from tf-teleport.sh output"
    echo "   # Then run: ./create-token.sh"
    echo ""
    echo "3. Deploy agents:"
    echo "   $0 apply --teleport-server-ip YOUR_IP --teleport-token YOUR_TOKEN"
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
    check-agents)
        check_agents
        ;;
    help-values)
        suggest_values
        ;;
    *)
        echo -e "${RED}❌ Error: Unknown command '$CMD'${NC}"
        echo ""
        if [[ "$CMD" == "plan" || "$CMD" == "apply" ]] && [[ "$ARGS" != *"--teleport-server-ip"* ]]; then
            suggest_values
        else
            show_usage
        fi
        exit 1
        ;;
esac 
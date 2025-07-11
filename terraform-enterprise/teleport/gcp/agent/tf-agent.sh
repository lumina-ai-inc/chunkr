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
    echo "  ssh [vm-number]                - SSH into an agent VM (defaults to VM 1)"
    echo "  debug-state                    - Debug terraform state detection"
    echo "  get-token                      - Get and display fresh token for testing"
    echo ""
    echo "Optional Variables:"
    echo "  --teleport-server-ip <ip>      - IP of the Teleport server (auto-retrieved)"
    echo "  --teleport-token <token>       - Token for agents to connect (auto-retrieved)"
    echo "  --teleport-ca-pin <pin>        - CA pin for server verification (auto-retrieved)"
    echo "  --agent-count <number>         - Number of agent VMs (default: 3)"
    echo "  --machine-type <type>          - VM machine type (default: n2-standard-2)"
    echo "  --disk-size <gb>               - Disk size in GB (default: 30)"
    echo "  --override-region <region>     - Override region"
    echo "  --override-zone <zone_suffix>  - Override zone suffix"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 plan                        # Automatically gets fresh teleport values"
    echo "  $0 apply --agent-count 5       # Automatically gets fresh teleport values"
    echo "  $0 plan --disk-size 1024       # Update disk size with fresh token"
    echo "  $0 apply --disk-size 1024      # Apply with fresh token"
    echo "  $0 ssh                         # SSH into agent VM 1"
    echo "  $0 ssh 2                       # SSH into agent VM 2"
    echo "  $0 get-token                   # Get and display fresh token for testing"
    echo "  $0 check-agents"
    echo "  $0 destroy"
}

# Global variables for parsed arguments
TELEPORT_SERVER_IP=""
TELEPORT_TOKEN=""
TELEPORT_CA_PIN=""
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
    TELEPORT_CA_PIN=""
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
            --teleport-ca-pin)
                TELEPORT_CA_PIN="$2"
                TF_ARGS="$TF_ARGS -var teleport_ca_pin=$2"
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



# Function to get fresh values from client
get_fresh_values() {
    echo -e "${BLUE}🔄 Getting fresh teleport values from client...${NC}"
    
    # Get server IP from client
    if [[ -z "$TELEPORT_SERVER_IP" ]]; then
        echo "  📡 Getting teleport server IP..."
        if [[ -f "../client/tf-teleport.sh" ]]; then
            local server_ip=$(cd ../client && ./tf-teleport.sh output 2>/dev/null | grep teleport_server_ip | sed 's/.*= "\(.*\)"/\1/')
            if [[ -n "$server_ip" ]]; then
                TELEPORT_SERVER_IP="$server_ip"
                TF_ARGS="$TF_ARGS -var teleport_server_ip=$server_ip"
                echo -e "${GREEN}  ✅ Got server IP: $server_ip${NC}"
            else
                echo -e "${RED}  ❌ Could not get server IP from client${NC}"
                return 1
            fi
        else
            echo -e "${RED}  ❌ Client terraform not found at ../client/${NC}"
            return 1
        fi
    fi
    
    # Get fresh token and ca-pin from server
    if [[ -z "$TELEPORT_TOKEN" ]] || [[ -z "$TELEPORT_CA_PIN" ]]; then
        echo "  🎫 Getting fresh teleport token and ca-pin..."
        echo -e "${YELLOW}  ⏳ SSHing to teleport server to get token and ca-pin...${NC}"
        
        # SSH to server and get token using Docker (includes ca-pin in output)
        local token_output=$(gcloud compute ssh teleport-server --zone us-central1-b --tunnel-through-iap --command "sudo docker exec teleport-server tctl tokens add --type=node" 2>/dev/null)
        
        # Extract token from token output (from the "The invite token:" line)
        local fresh_token=$(echo "$token_output" | grep -E 'The invite token: [a-f0-9]+' | sed 's/.*The invite token: \([a-f0-9]*\).*/\1/' | head -1)
        
        # Extract ca-pin from token output (from the --ca-pin= line)
        local ca_pin=$(echo "$token_output" | grep -E '\-\-ca-pin=sha256:[a-f0-9]+' | sed 's/.*--ca-pin=\(sha256:[a-f0-9]*\).*/\1/' | head -1)
        
        if [[ -n "$fresh_token" ]] && [[ -n "$ca_pin" ]]; then
            TELEPORT_TOKEN="$fresh_token"
            TELEPORT_CA_PIN="$ca_pin"
            TF_ARGS="$TF_ARGS -var teleport_token=$fresh_token -var teleport_ca_pin=$ca_pin"
            echo -e "${GREEN}  ✅ Got fresh token: ${fresh_token:0:8}...${NC}"
            echo -e "${GREEN}  ✅ Got ca-pin: ${ca_pin:0:20}...${NC}"
        else
            echo -e "${RED}  ❌ Could not get fresh token and ca-pin from server${NC}"
            echo -e "${YELLOW}  🔍 Debug: Token output was:${NC}"
            echo "$token_output"
            echo "  💡 Try manually: gcloud compute ssh teleport-server --zone us-central1-b --tunnel-through-iap"
            echo "     Then run: sudo docker exec teleport-server tctl tokens add --type=node"
            return 1
        fi
    fi
    
    echo -e "${GREEN}🎉 Successfully got fresh teleport values!${NC}"
    echo ""
}

# Function to get the join command for testing
get_test_token() {
    echo -e "${BLUE}🔑 Getting fresh token and ca-pin for testing...${NC}"
    echo -e "${YELLOW}  ⏳ SSHing to teleport server to get token and ca-pin...${NC}"
    
    # SSH to server and get token using Docker (includes ca-pin in output)
    local token_output=$(gcloud compute ssh teleport-server --zone us-central1-b --tunnel-through-iap --command "sudo docker exec teleport-server tctl tokens add --type=node" 2>/dev/null)
    
    echo ""
    echo -e "${BLUE}📄 Token output (includes ca-pin):${NC}"
    echo "$token_output"
    echo ""
    
    # Extract token and ca-pin
    local fresh_token=$(echo "$token_output" | grep -E 'The invite token: [a-f0-9]+' | sed 's/.*The invite token: \([a-f0-9]*\).*/\1/' | head -1)
    local ca_pin=$(echo "$token_output" | grep -E '\-\-ca-pin=sha256:[a-f0-9]+' | sed 's/.*--ca-pin=\(sha256:[a-f0-9]*\).*/\1/' | head -1)
    
    if [[ -n "$fresh_token" ]] && [[ -n "$ca_pin" ]]; then
        echo -e "${GREEN}✅ Fresh token and ca-pin retrieved successfully!${NC}"
        echo -e "${BLUE}Token: ${fresh_token}${NC}"
        echo -e "${BLUE}CA-Pin: ${ca_pin}${NC}"
        echo -e "${BLUE}Token length: ${#fresh_token} characters${NC}"
        echo ""
        echo -e "${YELLOW}💡 This token will expire in 30 minutes${NC}"
        echo ""
        echo -e "${BLUE}Full teleport start command:${NC}"
        echo "teleport start \\"
        echo "  --roles=node \\"
        echo "  --token=${fresh_token} \\"
        echo "  --ca-pin=${ca_pin} \\"
        echo "  --auth-server=TELEPORT_SERVER_IP:3025"
        return 0
    else
        echo -e "${RED}❌ Could not extract token and ca-pin from server response${NC}"
        return 1
    fi
}

# Function to validate required variables
validate_required() {
    local cmd=$1
    
    if [[ "$cmd" == "plan" || "$cmd" == "apply" ]]; then
        # If no teleport values provided, get fresh ones
        if [[ -z "$TELEPORT_SERVER_IP" ]] || [[ -z "$TELEPORT_TOKEN" ]] || [[ -z "$TELEPORT_CA_PIN" ]]; then
            if get_fresh_values; then
                echo -e "${GREEN}✅ Successfully obtained fresh teleport values${NC}"
            else
                echo -e "${RED}❌ Error: Could not automatically obtain teleport values${NC}"
                echo ""
                echo "Manual steps:"
                echo "1. Get server IP: cd ../client && ./tf-teleport.sh output"
                echo "2. Get token: gcloud compute ssh teleport-server --zone us-central1-b --tunnel-through-iap"
                echo "   Then run: ./create-token.sh"
                echo "3. Re-run with: $0 $cmd --teleport-server-ip <ip> --teleport-token <token> --teleport-ca-pin <pin>"
                exit 1
            fi
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

# Function to SSH into agent VM
ssh_into_agent() {
    local vm_number=${1:-1}
    
    echo -e "${BLUE}🔗 SSHing into agent VM ${vm_number}...${NC}"
    echo ""
    
    # Check if terraform is initialized and has state
    if ! terraform show -json > /dev/null 2>&1; then
        echo -e "${RED}❌ No infrastructure deployed${NC}"
        echo "Run: $0 apply"
        return 1
    fi
    
    # Get the VM name and zone
    local vm_name=$(terraform output -json agent_vm_names 2>/dev/null | jq -r ".[${vm_number}-1]" 2>/dev/null)
    local vm_zone=$(terraform show -json 2>/dev/null | jq -r ".values.root_module.resources[] | select(.type==\"google_compute_instance\" and .values.name==\"${vm_name}\") | .values.zone" 2>/dev/null)
    
    if [[ -z "$vm_name" || "$vm_name" == "null" ]]; then
        echo -e "${RED}❌ Agent VM ${vm_number} not found${NC}"
        echo ""
        echo "Available VMs:"
        terraform output -json agent_vm_names 2>/dev/null | jq -r 'to_entries[] | "  \(.key + 1): \(.value)"' 2>/dev/null || echo "  No VMs found"
        return 1
    fi
    
    if [[ -z "$vm_zone" || "$vm_zone" == "null" ]]; then
        echo -e "${RED}❌ Could not determine zone for VM ${vm_name}${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Found VM: ${vm_name} in zone: ${vm_zone}${NC}"
    echo -e "${BLUE}🚀 Connecting via IAP tunnel...${NC}"
    echo ""
    
    # Execute the SSH command
    gcloud compute ssh "${vm_name}" --zone "${vm_zone}" --tunnel-through-iap
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
    ssh)
        ssh_into_agent $ARGS
        ;;
    get-token)
        get_test_token
        ;;
    debug-state)
        echo "Debugging terraform state detection..."
        echo ""
        if [[ -d ".terraform" ]]; then
            echo -e "${GREEN}✅ Terraform initialized and state file found${NC}"
            echo "  - .terraform directory exists"
            echo "  - terraform.tfstate file found"
            echo "  - Checking for resources..."
            if terraform show 2>/dev/null | grep -q "google_compute_instance.agent_vms"; then
                echo -e "  ${GREEN}✅ google_compute_instance.agent_vms found${NC}"
            else
                echo -e "  ${RED}❌ google_compute_instance.agent_vms not found${NC}"
            fi
        else
            echo -e "${RED}❌ Terraform not initialized or state file not found${NC}"
            echo "  - .terraform directory does not exist"
            echo "  - terraform.tfstate file does not exist"
        fi
        echo ""
        ;;
    get-token)
        get_test_token
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
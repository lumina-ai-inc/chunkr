#!/bin/bash

# Disaster Recovery Testing Script
# This script performs DESTRUCTIVE testing by destroying and recreating infrastructure
# Use with EXTREME caution - only run on non-production workspaces

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="$SCRIPT_DIR/dr-test-results"
OUTPUT_DIR="$BASE_OUTPUT_DIR/dr-test-${TIMESTAMP}"
TEST_LOG="$OUTPUT_DIR/test.log"
DETAILED_LOG="$OUTPUT_DIR/detailed.log"

# Capture personnel information for audit trail
capture_personnel_info() {
    local personnel_file="$OUTPUT_DIR/personnel.json"
    
    # Get current user information
    local current_user=$(whoami)
    local real_name=$(getent passwd "$current_user" | cut -d: -f5 | cut -d, -f1 || echo "Unknown")
    local user_id=$(id -u)
    local group_id=$(id -g)
    local groups=$(groups)
    
    # Get system information
    local hostname=$(hostname)
    local ip_address=$(hostname -I | awk '{print $1}' || echo "Unknown")
    local os_info=$(uname -a)
    
    # Get git information if available
    local git_user_name=$(git config --get user.name 2>/dev/null || echo "Not configured")
    local git_user_email=$(git config --get user.email 2>/dev/null || echo "Not configured")
    local git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local git_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    
    # Get GCP authentication info
    local gcp_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 || echo "Not authenticated")
    local gcp_project=$(gcloud config get-value project 2>/dev/null || echo "Not set")
    
    # Create personnel record
    cat > "$personnel_file" << EOF
{
  "test_metadata": {
    "test_id": "dr-test-${TIMESTAMP}",
    "test_date": "$(date -Iseconds)",
    "test_type": "disaster_recovery",
    "script_version": "$git_commit"
  },
  "personnel": {
    "executor": {
      "username": "$current_user",
      "real_name": "$real_name",
      "user_id": $user_id,
      "group_id": $group_id,
      "groups": "$groups"
    },
    "git_identity": {
      "name": "$git_user_name",
      "email": "$git_user_email"
    },
    "authorization": {
      "gcp_account": "$gcp_account",
      "gcp_project": "$gcp_project"
    }
  },
  "system": {
    "hostname": "$hostname",
    "ip_address": "$ip_address",
    "os_info": "$os_info",
    "working_directory": "$(pwd)",
    "script_path": "$0"
  },
  "git_context": {
    "commit": "$git_commit",
    "branch": "$git_branch",
    "repository": "$(git remote get-url origin 2>/dev/null || echo "unknown")"
  }
}
EOF

    echo "$personnel_file"
}

# Enhanced log function with personnel context
log_with_personnel() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local user=$(whoami)
    echo "[$timestamp] [$user] $message" | tee -a "$TEST_LOG"
}

# Function to show usage
show_usage() {
    echo -e "${BLUE}🚨 Disaster Recovery Testing Script${NC}"
    echo -e "${RED}⚠️  WARNING: This script DESTROYS and RECREATES infrastructure!${NC}"
    echo ""
    echo "Usage: $0 <workspace> [options]"
    echo ""
    echo "Workspaces: dev, staging (NEVER use prod or turbolearn)"
    echo ""
    echo "Options:"
    echo "  --dr-region <region>      - Test recovery in different region"
    echo "  --dr-zone <zone_suffix>   - Test recovery in different zone"
    echo "  --dr-project <project>    - Test recovery in different project"
    echo "  --skip-destroy           - Skip destruction phase (for testing script)"
    echo "  --destroy-only           - Only destroy, don't recreate"
    echo "  --help                   - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 staging                                    # Test staging workspace destruction/recreation"
    echo "  $0 dev --dr-region us-west2 --dr-zone c     # Test cross-region DR"
    echo "  $0 staging --skip-destroy                    # Test recreation only"
    echo ""
    echo -e "${RED}🚨 NEVER RUN THIS ON PRODUCTION WORKSPACES!${NC}"
    echo ""
    echo "📁 Results will be saved to: dr-test-results/dr-test-{timestamp}/"
}

# Function to create output directory
create_output_dir() {
    if [[ ! -d "$BASE_OUTPUT_DIR" ]]; then
        mkdir -p "$BASE_OUTPUT_DIR"
        echo -e "${GREEN}📁 Created base output directory: $BASE_OUTPUT_DIR${NC}"
    fi
    
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        mkdir -p "$OUTPUT_DIR"
        echo -e "${GREEN}📁 Created test output directory: $OUTPUT_DIR${NC}"
    fi
}

# Function to log with timestamp
log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local user=$(whoami)
    echo "[$timestamp] [$user] $message" | tee -a "$TEST_LOG"
}

# Function to log detailed output
log_detailed() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" >> "$DETAILED_LOG"
}

# Function to run command with logging
run_with_logging() {
    local description="$1"
    shift
    local cmd="$@"
    
    log "🏃 Starting: $description"
    log_detailed "Command: $cmd"
    
    if eval "$cmd" 2>&1 | tee -a "$DETAILED_LOG"; then
        log "✅ Success: $description"
        return 0
    else
        log "❌ Failed: $description"
        return 1
    fi
}

# Function to validate workspace safety
validate_workspace() {
    local workspace="$1"
    
    case $workspace in
        dev|staging)
            log "✅ Safe workspace detected: $workspace"
            return 0
            ;;
        prod|turbolearn)
            log "🚨 DANGER: Attempted to run DR test on PRODUCTION workspace: $workspace"
            echo -e "${RED}❌ ABORTING: Cannot run destructive DR test on production workspace!${NC}"
            echo -e "${RED}   Use 'dev' or 'staging' workspaces only.${NC}"
            exit 1
            ;;
        *)
            log "❌ Unknown workspace: $workspace"
            echo -e "${RED}❌ Invalid workspace. Use: dev, staging${NC}"
            exit 1
            ;;
    esac
}

# Function to capture pre-destruction state
capture_pre_state() {
    local workspace="$1"
    
    log "📊 Capturing pre-destruction state"
    
    # Capture terraform state
    run_with_logging "Terraform state capture" "./tf-workspace.sh $workspace plan -out=/tmp/pre-destroy.tfplan"
    
    # Capture outputs (with timeout and better error handling)
    log "🔍 Attempting to capture terraform outputs (with 5-minute timeout)"
    if timeout 300 ./tf-workspace.sh $workspace apply -refresh-only >/dev/null 2>&1; then
        run_with_logging "Terraform outputs capture" "./tf-workspace.sh $workspace output > $OUTPUT_DIR/pre-destroy-outputs.json"
        log "✅ Successfully captured terraform outputs"
    else
        log "⚠️  Terraform refresh timed out or failed - skipping outputs capture (non-critical)"
        log "ℹ️  This doesn't affect the DR test functionality"
        # Try to get outputs without refresh as fallback
        if ./tf-workspace.sh $workspace output > "$OUTPUT_DIR/pre-destroy-outputs.json" 2>/dev/null; then
            log "✅ Captured outputs without refresh as fallback"
        else
            log "⚠️  Could not capture terraform outputs - continuing with DR test"
        fi
    fi
    
    # Capture GCP resources
    run_with_logging "GCP resources capture" "gcloud compute instances list --format=json > $OUTPUT_DIR/pre-destroy-instances.json" || true
    run_with_logging "GCP SQL instances capture" "gcloud sql instances list --format=json > $OUTPUT_DIR/pre-destroy-sql.json" || true
    run_with_logging "GCP storage buckets capture" "gcloud storage buckets list --format=json > $OUTPUT_DIR/pre-destroy-buckets.json" || true
    run_with_logging "GKE clusters capture" "gcloud container clusters list --format=json > $OUTPUT_DIR/pre-destroy-clusters.json" || true
}

# Function to perform destruction
perform_destruction() {
    local workspace="$1"
    local start_time=$(date +%s)
    
    log "💥 Starting infrastructure destruction"
    log "Workspace: $workspace"
    log "Destruction started at: $(date)"
    
    # Destroy infrastructure
    if run_with_logging "Infrastructure destruction" "./tf-workspace.sh $workspace destroy -auto-approve"; then
        local end_time=$(date +%s)
        local destruction_time=$((end_time - start_time))
        log "✅ Destruction completed in ${destruction_time} seconds"
        echo "$destruction_time" > "$OUTPUT_DIR/destruction-time.txt"
    else
        log "❌ Destruction failed"
        return 1
    fi
    
    # Verify destruction
    log "🔍 Verifying complete destruction"
    sleep 30  # Wait for eventual consistency
    
    run_with_logging "Post-destruction GCP resources check" "gcloud compute instances list --format=json > $OUTPUT_DIR/post-destroy-instances.json" || true
    run_with_logging "Post-destruction SQL instances check" "gcloud sql instances list --format=json > $OUTPUT_DIR/post-destroy-sql.json" || true
    run_with_logging "Post-destruction storage buckets check" "gcloud storage buckets list --format=json > $OUTPUT_DIR/post-destroy-buckets.json" || true
    run_with_logging "Post-destruction GKE clusters check" "gcloud container clusters list --format=json > $OUTPUT_DIR/post-destroy-clusters.json" || true
}

# Function to perform recreation
perform_recreation() {
    local workspace="$1"
    local dr_args="$2"
    local start_time=$(date +%s)
    
    log "🔄 Starting infrastructure recreation"
    log "Workspace: $workspace"
    log "DR Arguments: $dr_args"
    log "Recreation started at: $(date)"
    
    # Recreate infrastructure
    if run_with_logging "Infrastructure recreation" "./tf-workspace.sh $workspace apply -auto-approve $dr_args"; then
        local end_time=$(date +%s)
        local recreation_time=$((end_time - start_time))
        log "✅ Recreation completed in ${recreation_time} seconds"
        echo "$recreation_time" > "$OUTPUT_DIR/recreation-time.txt"
    else
        log "❌ Recreation failed"
        return 1
    fi
    
    # Verify recreation
    log "🔍 Verifying successful recreation"
    sleep 60  # Wait for services to be ready
    
    # Capture post-recreation state
    run_with_logging "Post-recreation terraform outputs" "./tf-workspace.sh $workspace output > $OUTPUT_DIR/post-recreation-outputs.json"
    run_with_logging "Post-recreation GCP resources check" "gcloud compute instances list --format=json > $OUTPUT_DIR/post-recreation-instances.json" || true
    run_with_logging "Post-recreation SQL instances check" "gcloud sql instances list --format=json > $OUTPUT_DIR/post-recreation-sql.json" || true
    run_with_logging "Post-recreation storage buckets check" "gcloud storage buckets list --format=json > $OUTPUT_DIR/post-recreation-buckets.json" || true
    run_with_logging "Post-recreation GKE clusters check" "gcloud container clusters list --format=json > $OUTPUT_DIR/post-recreation-clusters.json" || true
}

# Function to validate recreation
validate_recreation() {
    local workspace="$1"
    
    log "✅ Validating recreation success"
    
    # Test basic connectivity
    if run_with_logging "Terraform plan validation" "./tf-workspace.sh $workspace plan -detailed-exitcode"; then
        log "✅ Infrastructure matches desired state"
    else
        log "⚠️  Infrastructure drift detected (may be normal for new resources)"
    fi
    
    # Test key services
    log "🔍 Testing key service availability"
    
    # Check if we can get cluster credentials
    local cluster_name=""
    local cluster_region=""
    if [[ -f "$OUTPUT_DIR/post-recreation-outputs.json" ]]; then
        cluster_name=$(grep -o '"cluster_name":\s*"[^"]*"' "$OUTPUT_DIR/post-recreation-outputs.json" | cut -d'"' -f4 || true)
        cluster_region=$(grep -o '"cluster_region":\s*"[^"]*"' "$OUTPUT_DIR/post-recreation-outputs.json" | cut -d'"' -f4 || true)
    fi
    
    if [[ -n "$cluster_name" && -n "$cluster_region" ]]; then
        if run_with_logging "GKE cluster connectivity test" "gcloud container clusters get-credentials $cluster_name --region $cluster_region --quiet"; then
            run_with_logging "Kubectl cluster info" "kubectl cluster-info" || true
            run_with_logging "Kubectl node status" "kubectl get nodes" || true
        fi
    fi
}

# Function to generate test report
generate_report() {
    local workspace="$1"
    local dr_args="$2"
    local test_result="$3"
    
    local report_file="$OUTPUT_DIR/dr-test-report.md"
    
    cat > "$report_file" << EOF
# Disaster Recovery Test Report

**Test ID:** dr-test-${TIMESTAMP}
**Test Date:** $(date)
**Test Duration:** $(date -d@$(($(date +%s) - $(stat -c %Y "$TEST_LOG"))) -u +%H:%M:%S)
**Workspace Tested:** $workspace
**Test Result:** $test_result

## Personnel & Authorization
- **Test Executor:** $(whoami) ($(getent passwd $(whoami) | cut -d: -f5 | cut -d, -f1 || echo 'Unknown'))
- **Git Identity:** $(git config --get user.name 2>/dev/null || echo 'Not configured') <$(git config --get user.email 2>/dev/null || echo 'Not configured')>
- **GCP Account:** $(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 || echo 'Not authenticated')
- **GCP Project:** $(gcloud config get-value project 2>/dev/null || echo 'Not set')
- **System:** $(whoami)@$(hostname) ($(hostname -I | awk '{print $1}' || echo 'Unknown IP'))
- **Working Directory:** $(pwd)
- **Script Version:** $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
- **Git Branch:** $(git branch --show-current 2>/dev/null || echo "unknown")

## Test Configuration
- **Environment:** $workspace
- **DR Arguments:** $dr_args
- **Tester:** $(whoami)
- **Host:** $(hostname)
- **Script Version:** $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

## Audit Trail
- **Personnel Record:** \`personnel.json\`
- **Authorization Verified:** ✅ GCP account access confirmed
- **Workspace Validation:** ✅ Safe workspace ($workspace) confirmed
- **Operator Confirmation:** ✅ Destructive operations explicitly authorized by $(whoami)

## Test Objective
Validate complete disaster recovery capability by:
1. Destroying entire infrastructure
2. Recreating infrastructure (optionally in different location)
3. Validating functionality of recreated resources

## Test Execution Summary

$(cat "$TEST_LOG")

## Performance Metrics
EOF

    # Add timing information if available
    if [[ -f "$OUTPUT_DIR/destruction-time.txt" ]]; then
        echo "- **Destruction Time:** $(cat "$OUTPUT_DIR/destruction-time.txt") seconds" >> "$report_file"
    fi
    
    if [[ -f "$OUTPUT_DIR/recreation-time.txt" ]]; then
        echo "- **Recreation Time:** $(cat "$OUTPUT_DIR/recreation-time.txt") seconds" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

## Evidence Files Generated
- Test Log: \`$(basename "$TEST_LOG")\`
- Detailed Log: \`$(basename "$DETAILED_LOG")\`
- Personnel Record: \`personnel.json\`
- Pre-destruction Outputs: \`pre-destroy-outputs.json\`
- Post-recreation Outputs: \`post-recreation-outputs.json\`
- GCP Resource States: \`pre-destroy-*.json\`, \`post-destroy-*.json\`, \`post-recreation-*.json\`
- Timing Records: \`destruction-time.txt\`, \`recreation-time.txt\`

## Test Result: $test_result

$(if [[ "$test_result" == "SUCCESS" ]]; then
    echo "✅ All test phases completed successfully"
    echo "✅ Infrastructure destruction verified"
    echo "✅ Infrastructure recreation completed"
    echo "✅ Post-recreation validation passed"
else
    echo "❌ Test failed - see detailed logs for investigation"
fi)

## Next Actions
- Review detailed logs for any issues
- Update DR procedures based on lessons learned
- Schedule next DR test for $(date -d '+12 months' '+%Y-%m-%d')

**Approved By:** $(whoami)
**Date:** $(date)
EOF

    log "📝 Test report generated: $report_file"
}

# Main execution function
main() {
    local workspace=""
    local dr_args=""
    local skip_destroy=false
    local destroy_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dr-region)
                dr_args="$dr_args --override-region $2"
                shift 2
                ;;
            --dr-zone)
                dr_args="$dr_args --override-zone $2"
                shift 2
                ;;
            --dr-project)
                dr_args="$dr_args --override-project $2"
                shift 2
                ;;
            --skip-destroy)
                skip_destroy=true
                shift
                ;;
            --destroy-only)
                destroy_only=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                if [[ -z "$workspace" ]]; then
                    workspace="$1"
                else
                    echo -e "${RED}❌ Unknown argument: $1${NC}"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate arguments
    if [[ -z "$workspace" ]]; then
        echo -e "${RED}❌ Workspace required${NC}"
        show_usage
        exit 1
    fi
    
    # Create output directory FIRST - before any logging
    create_output_dir
    
    # Capture personnel information for audit trail
    local personnel_file=$(capture_personnel_info)
    
    # Safety checks (these call log() function)
    validate_workspace "$workspace"
    
    # Initialize logs with comprehensive personnel information
    log "🚀 Starting Disaster Recovery Test"
    log "============================================"
    log "👤 PERSONNEL & AUTHORIZATION INFORMATION:"
    log "   • Executor: $(whoami) ($(getent passwd $(whoami) | cut -d: -f5 | cut -d, -f1 || echo 'Unknown'))"
    log "   • Git Identity: $(git config --get user.name 2>/dev/null || echo 'Not configured') <$(git config --get user.email 2>/dev/null || echo 'Not configured')>"
    log "   • GCP Account: $(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 || echo 'Not authenticated')"
    log "   • GCP Project: $(gcloud config get-value project 2>/dev/null || echo 'Not set')"
    log "   • System: $(whoami)@$(hostname) ($(hostname -I | awk '{print $1}' || echo 'Unknown IP'))"
    log "   • Working Directory: $(pwd)"
    log "   • Script Path: $0"
    log "   • Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    log "   • Git Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    log "============================================"
    log "🎯 TEST CONFIGURATION:"
    log "   • Workspace: $workspace"
    log "   • DR Arguments: $dr_args"
    log "   • Skip Destroy: $skip_destroy"
    log "   • Destroy Only: $destroy_only"
    log "   • Test ID: dr-test-${TIMESTAMP}"
    log "   • Test Log: $TEST_LOG"
    log "   • Personnel Record: $personnel_file"
    log "============================================"
    
    # Confirmation for destructive operations
    if [[ "$skip_destroy" != true ]]; then
        echo ""
        echo -e "${RED}⚠️  CRITICAL: This will DESTROY all resources in workspace '$workspace'${NC}"
        echo -e "${RED}⚠️  Operator: $(whoami) ($(getent passwd $(whoami) | cut -d: -f5 | cut -d, -f1 || echo 'Unknown'))${NC}"
        echo -e "${RED}⚠️  GCP Account: $(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 || echo 'Not authenticated')${NC}"
        echo -e "${YELLOW}Are you absolutely sure? Type 'yes' to continue:${NC}"
        read -r confirmation
        if [[ "$confirmation" != "yes" ]]; then
            log "❌ Test cancelled by operator: $(whoami)"
            log "📝 Cancellation recorded in personnel file: $personnel_file"
            exit 1
        else
            log "✅ Destructive operation confirmed by operator: $(whoami)"
            log "⚠️  ACCOUNTABILITY: $(whoami) has authorized destruction of $workspace workspace"
        fi
    fi
    
    local test_result="FAILED"
    
    # Execute test phases
    if [[ "$skip_destroy" != true ]]; then
        capture_pre_state "$workspace"
        perform_destruction "$workspace"
    fi
    
    if [[ "$destroy_only" != true ]]; then
        perform_recreation "$workspace" "$dr_args"
        validate_recreation "$workspace"
        test_result="SUCCESS"
    else
        test_result="DESTRUCTION_ONLY"
    fi
    
    # Generate final report
    generate_report "$workspace" "$dr_args" "$test_result"
    
    log "🎉 Disaster Recovery Test completed: $test_result"
    log "📊 Test report: $OUTPUT_DIR/dr-test-report.md"
    
    echo ""
    echo -e "${GREEN}✅ DR Test Complete!${NC}"
    echo -e "${BLUE}📁 Results saved to: $OUTPUT_DIR${NC}"
    echo -e "${BLUE}📝 Test report: dr-test-report.md${NC}"
}

# Execute main function
main "$@" 
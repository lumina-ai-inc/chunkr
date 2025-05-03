#!/bin/bash

# Script to build and push Docker images to Azure Container Registry in parallel using tmux

# Required parameters
VERSION=$1
ACR_NAME=$2
RESOURCE_GROUP=$3

# Validate parameters
if [ -z "$VERSION" ] || [ -z "$ACR_NAME" ] || [ -z "$RESOURCE_GROUP" ]; then
    echo "Usage: $0 <version> <acr_name> <resource_group>"
    echo "Example: $0 1.0.0 myacr myresourcegroup"
    exit 1
fi

# Check if tmux is installed
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed. Please install it first."
    exit 1
fi

# Components to build from release-please.yml
ALL_COMPONENTS=(
    "server-azure"
    "task-azure"
    "web"
)

# Function to display menu for component selection
select_components() {
    echo "Available components to build:"
    for i in "${!ALL_COMPONENTS[@]}"; do
        echo "  $((i + 1)). ${ALL_COMPONENTS[$i]}"
    done

    echo "Enter the numbers of components you want to build (space-separated, or 'all' for all components):"
    read -r selection

    if [[ "$selection" == "all" ]]; then
        COMPONENTS=("${ALL_COMPONENTS[@]}")
        return
    fi

    COMPONENTS=()
    for num in $selection; do
        if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#ALL_COMPONENTS[@]}" ]; then
            COMPONENTS+=("${ALL_COMPONENTS[$((num - 1))]}")
        fi
    done

    if [ ${#COMPONENTS[@]} -eq 0 ]; then
        echo "No valid components selected. Exiting."
        exit 1
    fi

    echo "Selected components to build:"
    for component in "${COMPONENTS[@]}"; do
        echo "  - $component"
    done
    echo
}

# Call the function to select components
select_components

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer -o tsv)
if [ -z "$ACR_LOGIN_SERVER" ]; then
    echo "Failed to get ACR login server. Check your ACR name and resource group."
    exit 1
fi

# Login to ACR
echo "Logging in to ACR..."
az acr login --name $ACR_NAME

# Create a unique session name
SESSION_NAME="acr-build-$(date +%s)"

# Start a new tmux session
tmux new-session -d -s "$SESSION_NAME" -n "build"

# Create panes for each selected component
for i in "${!COMPONENTS[@]}"; do
    component="${COMPONENTS[$i]}"
    if [ $i -gt 0 ]; then
        # Split the window vertically
        tmux split-window -t "$SESSION_NAME:0" -v
    fi

    # Send the build and push commands to the pane
    tmux send-keys -t "$SESSION_NAME:0.$i" "echo \"Building and pushing $component:$VERSION...\"" C-m
    tmux send-keys -t "$SESSION_NAME:0.$i" "docker build -t $ACR_LOGIN_SERVER/$component:$VERSION -t $ACR_LOGIN_SERVER/$component:latest -f ./docker/$component/Dockerfile ." C-m
    tmux send-keys -t "$SESSION_NAME:0.$i" "docker push $ACR_LOGIN_SERVER/$component:$VERSION" C-m
    tmux send-keys -t "$SESSION_NAME:0.$i" "docker push $ACR_LOGIN_SERVER/$component:latest" C-m
    tmux send-keys -t "$SESSION_NAME:0.$i" "echo \"$component:$VERSION pushed successfully\"" C-m
    # tmux send-keys -t "$SESSION_NAME:0.$i" "exit" C-m
done

# Arrange the panes in a tiled layout
tmux select-layout -t "$SESSION_NAME:0" tiled

# Attach to the session
tmux attach-session -t "$SESSION_NAME"

# Wait for all panes to finish
echo "Waiting for all build and push operations to complete..."
tmux wait-for -S "$SESSION_NAME"

echo "All selected images built and pushed to $ACR_LOGIN_SERVER"

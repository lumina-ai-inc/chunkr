#!/bin/bash
echo "Welcome to the docker builder!"
echo "------------------------"

# Get version from .release-please-manifest-enterprise.json root entry
if [ -f ".release-please-manifest-enterprise.json" ]; then
    VERSION=$(grep -o '"\.": "[^"]*"' .release-please-manifest-enterprise.json | cut -d'"' -f4)
else
    VERSION="0.0.0"
fi

# Get the git SHA for reference
GIT_SHA=$(git rev-parse --short HEAD)

echo "Version from manifest: $VERSION"
echo "Git SHA: $GIT_SHA"
echo "------------------------"

# Find all docker.sh files in subdirectories of ./docker
docker_scripts=($(find ./docker -name "docker.sh"))

# Check if any docker.sh files were found
if [ ${#docker_scripts[@]} -eq 0 ]; then
    echo "No docker.sh files found in ./docker subdirectories"
    exit 1
fi

# Array to store scripts that user wants to build
selected_scripts=()

# Check if --all parameter was passed
if [ "$1" == "--all" ]; then
    selected_scripts=("${docker_scripts[@]}")
else
    # Ask user about each script
    for script in "${docker_scripts[@]}"; do
        dir_name=$(basename $(dirname "$script"))
        read -p "Build $dir_name? (y/n): " choice
        if [[ $choice =~ ^[Yy]$ ]]; then
            selected_scripts+=("$script")
        fi
    done
fi

# Check if any scripts were selected
if [ ${#selected_scripts[@]} -eq 0 ]; then
    echo "No docker builds were selected"
    exit 0
fi

# Export version variables for docker scripts to use
export VERSION=$VERSION
export GIT_SHA=$GIT_SHA

# Start a new tmux session
tmux new-session -d -s docker_session

# For each selected docker.sh file, create a new pane and run the script
for i in "${!selected_scripts[@]}"; do
    if [ $i -ne 0 ]; then
        # Split the window horizontally for additional scripts
        tmux split-window -h
        # Adjust the layout to make all panes equal size
        tmux select-layout tiled
    fi
    
    # Send the command to run the docker.sh script
    tmux send-keys -t $i "cd $(dirname ${selected_scripts[$i]}) && ./docker.sh" C-m
done

# Attach to the tmux session
tmux attach-session -t docker_session


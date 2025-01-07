#!/bin/bash

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


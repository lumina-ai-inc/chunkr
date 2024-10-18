#!/bin/bash

# Find all docker.sh files in subdirectories of ./docker
docker_scripts=($(find ./docker -name "docker.sh"))

# Check if any docker.sh files were found
if [ ${#docker_scripts[@]} -eq 0 ]; then
    echo "No docker.sh files found in ./docker subdirectories"
    exit 1
fi

# Start a new tmux session
tmux new-session -d -s docker_session

# For each docker.sh file, create a new pane and run the script
for i in "${!docker_scripts[@]}"; do
    if [ $i -ne 0 ]; then
        # Split the window horizontally for additional scripts
        tmux split-window -h
        # Adjust the layout to make all panes equal size
        tmux select-layout tiled
    fi
    
    # Send the command to run the docker.sh script
    tmux send-keys -t $i "cd $(dirname ${docker_scripts[$i]}) && ./docker.sh" C-m
done

# Attach to the tmux session
tmux attach-session -t docker_session


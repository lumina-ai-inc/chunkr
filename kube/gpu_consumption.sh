#!/bin/bash

gpu_nodes=$(kubectl get nodes -l node_pool=gpu-time-sharing -o jsonpath='{.items[*].metadata.name}')

tmux new-session -d -s gpu-info

window=0
for node in $gpu_nodes; do
    if [ $window -ne 0 ]; then
        tmux new-window -t gpu-info
    fi
    
    tmux rename-window -t gpu-info:$window "$node"
    
    # Get all pods on the node and check if any exist
    pods=$(kubectl get pods -n chunkmydocs --field-selector spec.nodeName=$node -o name)
    
    if [ -z "$pods" ]; then
        tmux send-keys -t gpu-info:$window "echo 'No GPU pod found on node $node'" C-m
    else
        # Get the first pod name, stripping the "pod/" prefix
        pod=$(echo "$pods" | head -n 1 | sed 's/^pod\///')
        tmux send-keys -t gpu-info:$window "echo 'Node: $node, Pod: $pod'" C-m
        tmux send-keys -t gpu-info:$window "kubectl exec -it $pod -n chunkmydocs -- watch -n 0.5 nvidia-smi" C-m
    fi
    
    window=$((window + 1))
done

tmux select-window -t gpu-info:0
tmux attach-session -t gpu-info

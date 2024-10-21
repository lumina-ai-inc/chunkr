#!/bin/bash

gpu_nodes=$(kubectl get nodes -l node_pool=gpu-time-sharing -o jsonpath='{.items[*].metadata.name}')

tmux new-session -d -s gpu-info

window=0
for node in $gpu_nodes; do
    if [ $window -ne 0 ]; then
        tmux split-window -h
    fi
    
    pod=$(kubectl get pods -n chunkmydocs --field-selector spec.nodeName=$node -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$pod" ]; then
        tmux send-keys -t gpu-info:0.$window "echo 'No GPU pod found on node $node'" C-m
    else
        tmux send-keys -t gpu-info:0.$window "echo 'Node: $node, Pod: $pod'" C-m
        tmux send-keys -t gpu-info:0.$window "kubectl exec -it $pod -n chunkmydocs -- watch -n 0.5 nvidia-smi" C-m
    fi
    
    window=$((window + 1))
done

tmux select-layout tiled
tmux attach-session -t gpu-info
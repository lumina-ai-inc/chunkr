#!/bin/bash

nodes=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}')

for node in $nodes
do
    echo "Draining node: $node"
    kubectl drain $node --ignore-daemonsets --delete-emptydir-data
    
    echo "Uncordoning node: $node"
    kubectl uncordon $node
done

echo "Pod redistribution complete"
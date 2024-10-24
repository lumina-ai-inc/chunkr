#!/bin/bash

evict_and_wait_pod() {
    local pod="$1"
    local namespace="$2"
    local wait_time="$3"

    echo "Evicting pod: $pod"
    kubectl delete pod "$pod" -n "$namespace"

    echo "Waiting for $pod to be rescheduled..."
    kubectl wait --for=condition=ready pod -l app="$pod" -n "$namespace" --timeout="${wait_time}s"

    echo "Pod $pod has been rescheduled and is running"
    echo "---"
}

# Export the function so it's available to subprocesses
export -f evict_and_wait_pod

redistribute_pods() {
    local namespace="${1:-chunkmydocs}"
    local wait_time=60  # Time to wait for pod to be rescheduled (in seconds)
    local temp_file=$(mktemp)

    # Get all pods in the namespace and save to temp file
    kubectl get pods -n "$namespace" -o custom-columns=":metadata.name,:spec.nodeName" \
            | grep "gpu" \
            | awk '{print $1}' > "$temp_file"

    # Use GNU Parallel to process pods concurrently
    parallel -a "$temp_file" evict_and_wait_pod {} "$namespace" "$wait_time"

    # Clean up
    rm "$temp_file"

    echo "All pods have been redistributed"
}

redistribute_pods
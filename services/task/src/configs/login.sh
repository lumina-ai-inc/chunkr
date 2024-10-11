#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <aws_access_key_id> <aws_secret_access_key> <region>"
    exit 1
fi

# Assign arguments to variables
AWS_ACCESS_KEY_ID=$1
AWS_SECRET_ACCESS_KEY=$2
AWS_DEFAULT_REGION=$3

# Configure AWS CLI
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_DEFAULT_REGION

# Verify the configuration
if aws sts get-caller-identity &>/dev/null; then
    echo "AWS CLI configured and login successful."
else
    echo "AWS CLI configuration failed or login unsuccessful."
    exit 1
fi


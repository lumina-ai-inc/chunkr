#!/bin/bash

# Get the commit message as a command line argument
m=$1

git add -A
git commit -m "$m"
git pull
git push
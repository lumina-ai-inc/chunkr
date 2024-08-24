#!/bin/bash

## USAGE ./pr-branch.sh cd/pr-branch "added pr-branch.sh"

b=$1
m=$2

git add -A
git commit -m "$m"
git checkout -b $b 
[ -z "$m" ] || git commit -m $m
gh pr create

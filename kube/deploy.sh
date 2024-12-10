#!/bin/bash
set -a
source .env
set +a

helm install chunkr ./chunkr-chart
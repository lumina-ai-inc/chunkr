# Get list of subnets with their regions and enable flow logs
gcloud compute networks subnets list --format="csv[no-heading](name,region)" | while IFS=',' read -r subnet region; do
  echo "Enabling flow logs for subnet: $subnet in region: $region"
  gcloud compute networks subnets update "$subnet" \
    --region="$region" \
    --enable-flow-logs \
    --logging-aggregation-interval=INTERVAL_10_MIN \
    --logging-flow-sampling=0.5 \
    --logging-metadata=INCLUDE_ALL
done

# Delete all SSH rules that allow 0.0.0.0/0 access
gcloud compute firewall-rules delete \
  allow-ssh-nourdine \
  chunkmydocs-dev-2-allow-ssh \
  chunkmydocs-prod-allow-ssh \
  chunkr-dev-allow-ssh \
  chunkr-prod-allow-ssh \
  dev-allow-ssh \
  dev-env-ishaan-allow-ssh \
  dev-gpu-allow-ssh \
  --quiet
#!/usr/bin/env bash

API_KEY=$(grep API_KEY .env | cut -d '=' -f2)

curl -H "Authorization: Bearer $API_KEY" "http://localhost:8000/pages-per-day?start=2024-12-01T00:00:00Z&end=2025-01-02T00:00:00Z"
echo

curl -H "Authorization: Bearer $API_KEY" "http://localhost:8000/status-breakdown?start=2024-12-01T00:00:00Z&end=2025-01-02T00:00:00Z"
echo

curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" \
    -d '{"start":"2024-12-01T00:00:00Z","end":"2025-01-02T00:00:00Z","limit":5}' \
    http://localhost:8000/top-users
echo

curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" \
    -d '{"email":"f@f.com","start":"2024-12-01T00:00:00Z","end":"2025-01-02T00:00:00Z"}' \
    http://localhost:8000/user-info
echo

curl -H "Authorization: Bearer $API_KEY" "http://localhost:8000/task-details?start=2024-12-01T00:00:00Z&end=2025-01-02T00:00:00Z"
echo

curl -H "Authorization: Bearer $API_KEY" "http://localhost:8000/lifetime-pages"
echo
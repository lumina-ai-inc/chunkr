#!/bin/bash
# Update existing PG__URL and REDIS__URL to use localhost

echo "🔧 Updating .env to use localhost for worker..."

# Backup
cp .env .env.backup2

# Replace Docker service names with localhost
sed -i '' 's|PG__URL=postgresql://postgres:postgres@postgres:5432/chunkr|PG__URL=postgresql://postgres:postgres@localhost:5432/orin|g' .env
sed -i '' 's|REDIS__URL=redis://redis:6379|REDIS__URL=redis://localhost:6379|g' .env

echo "✅ Updated PG__URL and REDIS__URL to use localhost"
echo ""
echo "📝 Backed up to: .env.backup2"
echo ""
echo "Verify changes:"
grep -E "PG__URL|REDIS__URL" .env

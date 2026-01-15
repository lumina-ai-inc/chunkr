#!/bin/bash
# Script to add localhost-compatible environment variables to .env

echo "🔧 Adding localhost configurations to .env for local worker..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    exit 1
fi

# Backup existing .env
cp .env .env.backup
echo "✅ Backed up .env to .env.backup"

# Add localhost configurations if they don't exist
if ! grep -q "DATABASE_URL=postgresql://postgres:postgres@localhost" .env; then
    echo "" >> .env
    echo "# Local worker configurations (uses localhost instead of Docker service names)" >> .env
    echo "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/orin" >> .env
    echo "✅ Added DATABASE_URL with localhost"
fi

if ! grep -q "REDIS_URL=redis://localhost" .env; then
    echo "REDIS_URL=redis://localhost:6379" >> .env
    echo "✅ Added REDIS_URL with localhost"
fi

if ! grep -q "AWS_S3_ENDPOINT=http://localhost:9000" .env; then
    echo "AWS_S3_ENDPOINT=http://localhost:9000" >> .env
    echo "✅ Added AWS_S3_ENDPOINT with localhost"
fi

if ! grep -q "AWS_S3_BUCKET" .env; then
    echo "AWS_S3_BUCKET=orin-documents" >> .env
    echo "✅ Added AWS_S3_BUCKET"
fi

if ! grep -q "AWS_ACCESS_KEY_ID" .env; then
    echo "AWS_ACCESS_KEY_ID=minioadmin" >> .env
    echo "AWS_SECRET_ACCESS_KEY=minioadmin" >> .env
    echo "✅ Added AWS credentials"
fi

# Check for Azure credentials
if ! grep -q "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" .env; then
    echo "" >> .env
    echo "# Azure Document Intelligence (for OCR)" >> .env
    echo "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=" >> .env
    echo "AZURE_DOCUMENT_INTELLIGENCE_KEY=" >> .env
    echo "⚠️  Added Azure placeholders - YOU NEED TO ADD YOUR CREDENTIALS!"
fi

echo ""
echo "✅ .env file updated!"
echo ""
echo "⚠️  IMPORTANT: Make sure to add your Azure credentials:"
echo "   AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/"
echo "   AZURE_DOCUMENT_INTELLIGENCE_KEY=your-api-key-here"
echo ""
echo "📝 Original file backed up to: .env.backup"



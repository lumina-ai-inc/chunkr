#!/bin/bash
# Script to update .env file for Docling OCR integration
# Run this script to configure environment variables for Docling

set -e

ENV_FILE=".env"

echo "🔧 Updating environment configuration for Docling OCR..."

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: .env file not found!"
    exit 1
fi

# Backup .env
cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
echo "✅ Created backup of .env"

# Update or add OCR_PROVIDER
if grep -q "^OCR_PROVIDER=" "$ENV_FILE"; then
    sed -i.tmp 's/^OCR_PROVIDER=.*/OCR_PROVIDER=docling/' "$ENV_FILE"
    rm -f "${ENV_FILE}.tmp"
    echo "✅ Updated OCR_PROVIDER=docling"
else
    echo "" >> "$ENV_FILE"
    echo "# Docling OCR Configuration" >> "$ENV_FILE"
    echo "OCR_PROVIDER=docling" >> "$ENV_FILE"
    echo "✅ Added OCR_PROVIDER=docling"
fi

# Add DOCLING_SERVICE_URL if not present
if ! grep -q "^DOCLING_SERVICE_URL=" "$ENV_FILE"; then
    echo "DOCLING_SERVICE_URL=http://docling-service:8000" >> "$ENV_FILE"
    echo "✅ Added DOCLING_SERVICE_URL"
fi

echo ""
echo "✅ Environment configuration updated successfully!"
echo ""
echo "📋 Summary of changes:"
echo "  - OCR_PROVIDER=docling"
echo "  - DOCLING_SERVICE_URL=http://docling-service:8000"
echo ""
echo "💡 Next steps:"
echo "  1. Review the changes in .env"
echo "  2. Run: make stop && docker image prune -a"
echo "  3. Run: make start"
echo "  4. Test document upload"



#!/bin/bash
# Setup script for PDF conversion tools (LibreOffice + ImageMagick)
# Required for DOCX/XLSX → PDF conversion for document preview

set -e

echo "🔍 Checking PDF conversion tools..."

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only. For Linux, run:"
    echo "   sudo apt-get install libreoffice-writer libreoffice-calc imagemagick"
    exit 1
fi

# Check Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Install from: https://brew.sh"
    exit 1
fi

echo "✅ Homebrew found"

# Check LibreOffice
if command -v soffice &> /dev/null; then
    echo "✅ LibreOffice already installed: $(soffice --version)"
else
    echo "📦 Installing LibreOffice..."
    brew install --cask libreoffice
    echo "✅ LibreOffice installed"
fi

# Check ImageMagick
if command -v convert &> /dev/null; then
    echo "✅ ImageMagick already installed: $(convert --version | head -1)"
else
    echo "📦 Installing ImageMagick..."
    brew install imagemagick
    echo "✅ ImageMagick installed"
fi

echo ""
echo "🎉 PDF conversion tools are ready!"
echo "💡 You can now restart the deal worker to enable DOCX/XLSX preview"

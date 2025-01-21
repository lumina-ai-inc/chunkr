# Chunkr AI Node.js Client

[![npm version](https://badge.fury.io/js/chunkr-ai.svg)](https://badge.fury.io/js/chunkr-ai)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Official Node.js client for the [Chunkr API](https://chunkr.ai) - A powerful document processing and chunking service for RAG applications.

## Features

- 📄 Process documents (PDF, Word, etc.)
- 🔍 OCR support with configurable strategies
- 📊 Layout analysis and intelligent segmentation
- 🔄 Async task management with polling
- 📝 Multiple output formats (HTML, Markdown, JSON)
- 🛠️ Configurable processing options

## Installation

```bash
npm install chunkr-ai
```

## Quick Start

```typescript
import { Chunkr } from "chunkr-ai";

// Initialize client with API key
const chunkr = new Chunkr("your-api-key");

// Process a document
const task = await chunkr.upload("path/to/document.pdf");

// Access the processed content
console.log(task.getHtml()); // Get HTML output
console.log(task.getMarkdown()); // Get Markdown output
console.log(task.output); // Get full structured output
```

## Authentication

You can provide your API key in several ways:

```typescript
// 1. Direct initialization
const chunkr = new Chunkr("your-api-key");
// 2. Environment variable
// Set CHUNKR_API_KEY in your environment or .env file
const chunkr = new Chunkr();
// 3. Configuration object
const chunkr = new Chunkr({
  apiKey: "your-api-key",
  baseUrl: "https://api.chunkr.ai", // Optional custom API URL for version control
});
```

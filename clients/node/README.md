# Chunkr AI Node.js SDK

Official Node.js client for [Chunkr](https://chunkr.ai)

The Chunkr Node SDK converts complex documents into RAG/LLM-ready data. It provides a simple interface for using the Chunkr API.

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

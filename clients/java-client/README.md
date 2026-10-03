# Chunkr Java Client

Java client library for the [Chunkr](https://chunkr.ai) document intelligence API. Chunkr transforms complex documents (PDFs, PPTs, Word docs, images) into RAG/LLM-ready chunks with layout analysis, OCR, and semantic segmentation.

## Features

- 🔍 **Layout Analysis** - Intelligent document structure detection
- 📄 **OCR + Bounding Boxes** - Extract text with precise positioning
- 🔗 **Structured Output** - HTML, Markdown, and JSON formats
- 🎯 **VLM Processing Controls** - Fine-tune processing with Vision Language Models
- ⚡ **Async Processing** - Upload and poll for results
- 🛠️ **Multiple Input Types** - File paths, URLs, byte arrays, InputStreams

## Installation

### Maven

Add this dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>ai.chunkr</groupId>
    <artifactId>chunkr-java</artifactId>
    <version>0.1.0</version>
</dependency>
```

### Gradle

Add this dependency to your `build.gradle`:

```gradle
implementation 'ai.chunkr:chunkr-java:0.1.0'
```

## Quick Start

### 1. Get Your API Key

Sign up at [chunkr.ai](https://chunkr.ai) and get your API key.

### 2. Basic Usage

```java
import ai.chunkr.client.ChunkrClient;
import ai.chunkr.client.models.TaskResponse;

public class ChunkrExample {
    public static void main(String[] args) {
        // Initialize client with API key
        ChunkrClient client = new ChunkrClient("your-api-key-here");
        
        try {
            // Upload and process a document
            TaskResponse task = client.upload("path/to/document.pdf");
            
            // Get results in different formats
            String html = task.getHtml();
            String markdown = task.getMarkdown();
            String content = task.getContent();
            
            System.out.println("HTML: " + html);
            System.out.println("Markdown: " + markdown);
            
        } finally {
            // Clean up resources
            client.close();
        }
    }
}
```

### 3. Environment Variables

You can set your API key as an environment variable:

```bash
export CHUNKR_API_KEY=your-api-key-here
```

Then initialize the client without parameters:

```java
ChunkrClient client = new ChunkrClient(); // Uses CHUNKR_API_KEY
```

## Advanced Usage

### Configuration Options

```java
import ai.chunkr.client.models.Configuration;
import ai.chunkr.client.enums.OcrStrategy;
import ai.chunkr.client.enums.SegmentationStrategy;

Configuration config = Configuration.builder()
    .ocrStrategy(OcrStrategy.AUTO)
    .segmentationStrategy(SegmentationStrategy.LAYOUT_ANALYSIS)
    .highResolution(true)
    .expiresIn(3600) // 1 hour
    .build();

TaskResponse task = client.upload("document.pdf", config);
```

### Different Input Types

```java
// From file path
TaskResponse task1 = client.upload("document.pdf");

// From URL
TaskResponse task2 = client.upload("https://example.com/document.pdf");

// From byte array
byte[] documentBytes = Files.readAllBytes(Paths.get("document.pdf"));
TaskResponse task3 = client.upload(documentBytes);

// From InputStream
try (InputStream stream = new FileInputStream("document.pdf")) {
    TaskResponse task4 = client.upload(stream);
}

// From File object
File file = new File("document.pdf");
TaskResponse task5 = client.upload(file);
```

### Async Processing

```java
// Create task without waiting
TaskResponse task = client.createTask("document.pdf");

// Do other work...

// Poll for completion when ready
task.poll();

// Or poll with custom interval
task.poll(2000); // Poll every 2 seconds
```

### Task Management

```java
// Get task by ID
TaskResponse task = client.getTask("task-id");

// Update task configuration
Configuration newConfig = Configuration.builder()
    .ocrStrategy(OcrStrategy.ALL)
    .build();
TaskResponse updatedTask = client.updateTask("task-id", newConfig);

// Cancel task
client.cancelTask("task-id");

// Delete task
client.deleteTask("task-id");
```

### Processing Segments

```java
import ai.chunkr.client.models.Output;
import ai.chunkr.client.models.Chunk;
import ai.chunkr.client.models.Segment;

TaskResponse task = client.upload("document.pdf");
Output output = task.getOutput();

if (output != null && output.getChunks() != null) {
    for (Chunk chunk : output.getChunks()) {
        System.out.println("Chunk ID: " + chunk.getChunkId());
        
        for (Segment segment : chunk.getSegments()) {
            System.out.println("Segment Type: " + segment.getSegmentType());
            System.out.println("Content: " + segment.getContent());
            System.out.println("HTML: " + segment.getHtml());
            System.out.println("Markdown: " + segment.getMarkdown());
        }
    }
}
```

## Configuration

### Client Configuration

```java
import ai.chunkr.client.config.ClientConfig;

ClientConfig config = ClientConfig.builder()
    .apiKey("your-api-key")
    .baseUrl("https://api.chunkr.ai") // Optional: for self-hosted instances
    .build();

ChunkrClient client = new ChunkrClient(config);
```

### Processing Configuration

```java
import ai.chunkr.client.models.*;
import ai.chunkr.client.enums.*;

// Chunk processing settings
ChunkProcessing chunkProcessing = ChunkProcessing.builder()
    .targetLength(1000)
    .build();

// Generation settings for different content types
GenerationConfig textGenConfig = GenerationConfig.builder()
    .html(GenerationStrategy.AUTO)
    .markdown(GenerationStrategy.LLM)
    .llm("gpt-4o")
    .cropImage(CroppingStrategy.AUTO)
    .build();

// Segment processing for different element types
SegmentProcessing segmentProcessing = SegmentProcessing.builder()
    .text(textGenConfig)
    .table(textGenConfig)
    .title(textGenConfig)
    .build();

// Complete configuration
Configuration config = Configuration.builder()
    .chunkProcessing(chunkProcessing)
    .segmentProcessing(segmentProcessing)
    .ocrStrategy(OcrStrategy.AUTO)
    .segmentationStrategy(SegmentationStrategy.LAYOUT_ANALYSIS)
    .highResolution(true)
    .expiresIn(7200)
    .build();
```

## Error Handling

```java
try {
    TaskResponse task = client.upload("document.pdf");
    String content = task.getContent();
} catch (RuntimeException e) {
    System.err.println("Processing failed: " + e.getMessage());
    // Handle error appropriately
}
```

## Self-Hosted Deployment

For self-hosted Chunkr instances:

```java
ChunkrClient client = new ChunkrClient("your-api-key", "https://your-chunkr-instance.com");
```

Or using environment variables:

```bash
export CHUNKR_API_KEY=your-api-key
export CHUNKR_URL=https://your-chunkr-instance.com
```

## API Reference

### ChunkrClient

| Method | Description |
|--------|-------------|
| `upload(Object file)` | Upload file and wait for completion |
| `upload(Object file, Configuration config)` | Upload with configuration |
| `createTask(Object file)` | Create task without waiting |
| `createTask(Object file, Configuration config)` | Create task with configuration |
| `getTask(String taskId)` | Get task by ID |
| `updateTask(String taskId, Configuration config)` | Update task configuration |
| `deleteTask(String taskId)` | Delete task |
| `cancelTask(String taskId)` | Cancel task |
| `close()` | Close client and release resources |

### TaskResponse

| Method | Description |
|--------|-------------|
| `poll()` | Poll until completion (1s interval) |
| `poll(long intervalMs)` | Poll with custom interval |
| `getHtml()` | Get HTML content |
| `getMarkdown()` | Get Markdown content |
| `getContent()` | Get raw text content |
| `cancel()` | Cancel this task |
| `delete()` | Delete this task |

## Requirements

- Java 11 or higher
- Maven 3.6+ or Gradle 6.0+

## Dependencies

- OkHttp 4.11.0 - HTTP client
- Jackson 2.15.2 - JSON processing
- SLF4J 2.0.7 - Logging

## Support

- 📧 Email: [mehul@chunkr.ai](mailto:mehul@chunkr.ai)
- 🌐 Website: [chunkr.ai](https://chunkr.ai)
- 📖 Documentation: [docs.chunkr.ai](https://docs.chunkr.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/lumina-ai-inc/chunkr/issues)

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](../../LICENSE) file for details.

For commercial licensing options, contact us at [mehul@chunkr.ai](mailto:mehul@chunkr.ai).
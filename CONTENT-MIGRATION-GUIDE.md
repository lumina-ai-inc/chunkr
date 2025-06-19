# Content Generation Migration Guide

## Overview

This guide documents the consolidation of HTML and markdown generation in Chunkr. Previously, you could generate both HTML and markdown content simultaneously. Now, you choose a single format per segment type, streamlining content generation and improving performance.

## Key Changes

### Before (Legacy)
- Separate `html` and `markdown` strategy fields
- Could generate both formats simultaneously
- Content returned in both `segment.html` and `segment.markdown` fields
- `segment.content` contained OCR-extracted text

### After (New)
- Single `format` field choosing between `Html` or `Markdown`
- Single `strategy` field for generation approach
- Content returned in `segment.content` field matching chosen format
- `segment.text` contains OCR-extracted text (moved from `segment.content`)
- **Breaking Change**: `segment.content` purpose completely changed

## Migration Priority

### 🚨 **CRITICAL - Immediate Action Required**

#### 1. Pipeline.Chunkr Table Processing Default Change  
**Impact**: Tables no longer generate AI-enhanced markdown by default
```python
# ❌ BROKEN - Default config no longer generates AI markdown for tables
# Old behavior: Tables got both HTML and AI-enhanced markdown automatically

# ✅ FIXED - Explicitly configure AI table processing
Table=GenerationConfig(
    format=SegmentFormat.MARKDOWN,
    strategy=GenerationStrategy.LLM
)
```
**Who's affected**: Applications using `Pipeline.Chunkr` with default table config expecting AI-generated `segment.markdown`
**Who's not affected**: Applications using `Pipeline.Chunkr` with a explicit table config

### ⚠️ **MEDIUM - Action Recommended**

#### 2. Pipeline.Azure Table Structure Changes
**Impact**: Table structure output changes (not LLM-enhanced but still structured differently)
```python
# 📋 AFFECTED - Table structure may differ from previous versions
# Azure tables still work but output structure may have changed
```
**Who's affected**: Applications using `Pipeline.Azure` with default table config relying on specific table markdown/HTML structure
**Who's not affected**: Applications using `Pipeline.Azure` with a explicit table config

#### 3. OCR Text Access Breaking Change
**Impact**: Applications accessing OCR text from `segment.content` will break
```python
# ❌ BROKEN - Will now return HTML/Markdown instead of OCR text
ocr_text = segment.content  

# ✅ FIXED - Update to use segment.text
ocr_text = segment.text
```
**Who's affected**: Applications reading `segment.content` expecting OCR text (rare usage pattern)

### 💡 **LOW - Performance Optimization**

#### 4. Update to New Content Access Pattern
**Impact**: Performance improvement and future-proofing
```python
# 📊 OLD - Accessing format-specific fields
html_content = segment.html
markdown_content = segment.markdown

# ⚡ NEW - Access unified content field  
generated_content = segment.content  # Contains format you requested
```
**Who's affected**: All applications (optional upgrade for better performance)

### 📋 **Quick Self-Assessment**

**Am I affected by Critical Issue #1?**  
- [ ] Do I use `Pipeline.Chunkr` (layout analysis)?
- [ ] Do I process tables without explicit configuration?
- [ ] Do I expect `segment.markdown` to contain AI-enhanced table content?
- [ ] **If yes to all**: Add explicit table configuration immediately

**Am I affected by Medium Issue #2?**
- [ ] Do I use `Pipeline.Azure`?
- [ ] Do I rely on specific table HTML/markdown structure?
- [ ] **If yes to both**: Test and validate table output structure

**Am I affected by Medium Issue #3?**
- [ ] Do I access `segment.content` anywhere in my code?
- [ ] Do I expect `segment.content` to contain OCR text?
- [ ] **If yes to both**: Update to `segment.text`

## Python Client Migration

### Basic Configuration

**Legacy approach:**
```python
from chunkr_ai import Chunkr
from chunkr_ai.models import Configuration, SegmentProcessing, GenerationConfig, GenerationStrategy, EmbedSource

config = Configuration(
    segment_processing=SegmentProcessing(
        Page=GenerationConfig(
            html=GenerationStrategy.LLM,
            markdown=GenerationStrategy.LLM,
            embed_sources=[EmbedSource.MARKDOWN]
        )
    )
)
```

**New approach:**
```python
from chunkr_ai import Chunkr
from chunkr_ai.models import Configuration, SegmentProcessing, GenerationConfig, GenerationStrategy, EmbedSource, SegmentFormat

config = Configuration(
    segment_processing=SegmentProcessing(
        Page=GenerationConfig(
            format=SegmentFormat.MARKDOWN,
            strategy=GenerationStrategy.LLM,
            embed_sources=[EmbedSource.CONTENT]
        )
    )
)
```

### Multiple Segment Types

**Legacy approach:**
```python
config = Configuration(
    segment_processing=SegmentProcessing(
        Table=GenerationConfig(
            html=GenerationStrategy.AUTO,
            markdown=GenerationStrategy.LLM,
        ),
        Picture=GenerationConfig(
            html=GenerationStrategy.AUTO,
            markdown=GenerationStrategy.AUTO,
        )
    )
)
```

**New approach:**
```python
config = Configuration(
    segment_processing=SegmentProcessing(
        Table=GenerationConfig(
            format=SegmentFormat.HTML,
            strategy=GenerationStrategy.LLM,
        ),
        Picture=GenerationConfig(
            format=SegmentFormat.MARKDOWN,
            strategy=GenerationStrategy.AUTO,
        )
    )
)
```

### Embedding Sources

**Legacy approach:**
```python
# Separate sources for different formats
embed_sources=[EmbedSource.HTML, EmbedSource.MARKDOWN, EmbedSource.LLM]
```

**New approach:**
```python
# Single content source plus optional LLM
embed_sources=[EmbedSource.CONTENT, EmbedSource.LLM]
```

### Complete Example

**New comprehensive configuration:**
```python
from chunkr_ai import Chunkr
from chunkr_ai.models import (
    Configuration, 
    SegmentProcessing, 
    GenerationConfig, 
    GenerationStrategy,
    SegmentFormat,
    EmbedSource,
    CroppingStrategy
)

# Configure different formats for different segment types
config = Configuration(
    segment_processing=SegmentProcessing(
        Page=GenerationConfig(
            format=SegmentFormat.MARKDOWN,
            strategy=GenerationStrategy.LLM,
            llm="Summarize this page content clearly",
            embed_sources=[EmbedSource.CONTENT, EmbedSource.LLM],
            crop_image=CroppingStrategy.AUTO,
            extended_context=True
        ),
        Table=GenerationConfig(
            format=SegmentFormat.HTML,
            strategy=GenerationStrategy.LLM,
            embed_sources=[EmbedSource.CONTENT],
            crop_image=CroppingStrategy.ALL
        ),
        Picture=GenerationConfig(
            format=SegmentFormat.MARKDOWN,
            strategy=GenerationStrategy.AUTO,
            embed_sources=[EmbedSource.CONTENT]
        )
    )
)

chunkr = Chunkr(api_key="your-api-key")
task = chunkr.upload("document.pdf", configuration=config)

# Access content in the chosen format
for segment in task.segments:
    print(f"Generated Content: {segment.content}")  # Contains either HTML or Markdown
    print(f"OCR Text: {segment.text}")              # Contains raw OCR-extracted text
```

## Rust Core Changes

### Configuration Structs

**Legacy:**
```rust
pub struct GenerationConfig {
    pub html: GenerationStrategy,
    pub markdown: GenerationStrategy,
    pub llm: Option<String>,
    pub embed_sources: Vec<EmbedSource>,
}
```

**New:**
```rust
pub struct GenerationConfig {
    pub format: SegmentFormat,
    pub strategy: GenerationStrategy,
    pub llm: Option<String>,
    pub embed_sources: Vec<EmbedSource>,
    // Deprecated for backwards compatibility
    pub html: Option<GenerationStrategy>,
    pub markdown: Option<GenerationStrategy>,
}

pub enum SegmentFormat {
    Html,
    Markdown,
}
```

### Segment Structure

**Legacy:**
```rust
pub struct Segment {
    pub content: String,  // OCR text content
    pub html: String,
    pub markdown: String,
    // ...
}
```

**New:**
```rust
pub struct Segment {
    pub content: String,  // Generated content (HTML or Markdown)
    pub text: String,     // OCR text content
    pub html: String,     // Still available for backwards compatibility
    pub markdown: String, // Still available for backwards compatibility
    // ...
}
```

### Processing Logic

**New content generation:**
```rust
// Determine content based on chosen format
segment.content = match config.format {
    SegmentFormat::Html => segment.html.clone(),
    SegmentFormat::Markdown => segment.markdown.clone(),
};
```

## Backwards Compatibility

### Deserialization Process Explained

The system uses a sophisticated deserialization strategy to handle both legacy and new configuration formats seamlessly. Here's how it works:

#### Step 1: JSON Parsing

**Legacy Configuration Input:**
```json
{
  "segment_processing": {
    "page": {
      "html": "LLM",
      "markdown": "Auto",
      "llm": "Summarize this content",
      "embed_sources": ["HTML", "Markdown"]
    }
  }
}
```

**New Configuration Input:**
```json
{
  "segment_processing": {
    "page": {
      "format": "Markdown",
      "strategy": "LLM", 
      "llm": "Summarize this content",
      "embed_sources": ["Content"]
    }
  }
}
```

#### Step 2: Struct Deserialization

The Rust structs are designed to accept both formats:

```rust
#[derive(Deserialize)]
pub struct GenerationConfig {
    // New primary fields
    pub format: Option<SegmentFormat>,
    pub strategy: Option<GenerationStrategy>,
    
    // Legacy fields (still deserialized but deprecated)
    #[serde(default)]
    pub html: Option<GenerationStrategy>,
    #[serde(default)]
    pub markdown: Option<GenerationStrategy>,
    
    // Common fields
    pub llm: Option<String>,
    pub embed_sources: Option<Vec<EmbedSource>>,
    pub crop_image: Option<CroppingStrategy>,
    pub extended_context: Option<bool>,
}
```

#### Step 3: Field Resolution Logic

After deserialization, the system resolves the effective configuration using simple fallback logic:

```rust
impl GenerationConfig {
    // Simplified logic - new fields take precedence over deprecated ones
    pub fn get_effective_format(&self) -> SegmentFormat {
        // Use new format field if present
        if let Some(format) = &self.format {
            return format.clone();
        }
        
        // Fallback to deprecated fields
        // (Implementation details may vary, but concept is similar)
        if self.html.is_some() {
            SegmentFormat::Html
        } else if self.markdown.is_some() {
            SegmentFormat::Markdown
        } else {
            // Default varies by segment type
            SegmentFormat::Html  // or Markdown depending on segment
        }
    }
    
    pub fn get_effective_strategy(&self) -> GenerationStrategy {
        // Use new strategy field if present
        if let Some(strategy) = &self.strategy {
            return strategy.clone();
        }
        
        // Fallback to whichever deprecated field is being used
        self.html.unwrap_or(self.markdown.unwrap_or(GenerationStrategy::Auto))
    }
}
```

#### Step 4: EmbedSource Resolution

EmbedSources also require mapping for backwards compatibility:

```rust
// The actual mapping logic (simplified)
match embed_source {
    EmbedSource::Content => &segment.text,    // New approach - use OCR text
    EmbedSource::HTML => &segment.html,       // Deprecated but still works  
    EmbedSource::Markdown => &segment.markdown, // Deprecated but still works
    EmbedSource::LLM => &segment.llm,
}
```

#### Step 5: Runtime Processing

During processing, the system uses the resolved configuration:

```rust
async fn process_segment(segment: &mut Segment, config: &GenerationConfig) {
    let format = config.get_effective_format();
    let strategy = config.get_effective_strategy();
    
    // Generate content based on resolved format
    match format {
        SegmentFormat::Html => {
            segment.html = generate_html(segment, strategy).await?;
            segment.content = segment.html.clone(); // Key change: content = generated content
        },
        SegmentFormat::Markdown => {
            segment.markdown = generate_markdown(segment, strategy).await?;
            segment.content = segment.markdown.clone(); // Key change: content = generated content
        }
    }
    
    // Key change: OCR text moved to segment.text
    segment.text = extract_ocr_text(segment);
    
    // Backwards compatibility: still populate both html/markdown fields when possible
    if segment.html.is_empty() {
        segment.html = generate_basic_html(segment);
    }
    if segment.markdown.is_empty() {
        segment.markdown = generate_basic_markdown(segment);
    }
}
```

### Configuration Deserialization

The system handles both old and new configuration formats through a fallback mechanism:

**When using deprecated fields:**
```python
# Old configuration still works
GenerationConfig(
    html=GenerationStrategy.LLM,        # Deprecated
    markdown=GenerationStrategy.AUTO,   # Deprecated
)
```

**Deserialization logic:**
```rust
// Simplified logic in Rust core
impl GenerationConfig {
    pub fn get_effective_format_and_strategy(&self) -> (SegmentFormat, GenerationStrategy) {
        // New approach takes precedence
        if let Some(format) = &self.format {
            return (format.clone(), self.strategy.clone());
        }
        
        // Fallback to deprecated fields
        if self.html.is_some() && self.markdown.is_some() {
            // Both specified - prefer the LLM strategy
            if self.html == Some(GenerationStrategy::LLM) {
                return (SegmentFormat::Html, GenerationStrategy::LLM);
            } else if self.markdown == Some(GenerationStrategy::LLM) {
                return (SegmentFormat::Markdown, GenerationStrategy::LLM);
            }
        }
        
        // Single format specified
        if let Some(html_strategy) = &self.html {
            return (SegmentFormat::Html, html_strategy.clone());
        }
        if let Some(markdown_strategy) = &self.markdown {
            return (SegmentFormat::Markdown, markdown_strategy.clone());
        }
        
        // Default fallback based on segment type
        self.get_segment_default()
    }
}
```

### Response Field Population

The system populates multiple fields to maintain compatibility:

**Processing flow:**
```rust
// Simplified processing logic
async fn process_segment(segment: &mut Segment, config: &GenerationConfig) {
    let (format, strategy) = config.get_effective_format_and_strategy();
    
    // Generate content based on effective configuration
    match format {
        SegmentFormat::Html => {
            segment.html = generate_html_content(segment, strategy).await?;
            segment.content = segment.html.clone(); // New primary field
        },
        SegmentFormat::Markdown => {
            segment.markdown = generate_markdown_content(segment, strategy).await?;
            segment.content = segment.markdown.clone(); // New primary field
        }
    }
    
    // Always populate text field with OCR results
    segment.text = extract_ocr_text(segment);
    
    // For backwards compatibility, ensure both html/markdown are populated
    // even if only one was generated
    if segment.html.is_empty() && strategy == GenerationStrategy::Auto {
        segment.html = generate_basic_html(segment);
    }
    if segment.markdown.is_empty() && strategy == GenerationStrategy::Auto {
        segment.markdown = generate_basic_markdown(segment);
    }
}
```

### EmbedSource Compatibility

**Embed source mapping:**
```rust
// Handle deprecated embed sources
impl EmbedSource {
    pub fn get_effective_content(&self, segment: &Segment) -> String {
        match self {
            EmbedSource::Content => segment.text.clone(), // New approach
            EmbedSource::HTML => segment.html.clone(),    // Deprecated but works
            EmbedSource::Markdown => segment.markdown.clone(), // Deprecated but works
            EmbedSource::LLM => segment.llm.clone().unwrap_or_default(),
        }
    }
}
```

### What Happens with Old Configurations

**Scenario 1: Legacy config with both HTML and Markdown**
```python
# Old config
GenerationConfig(html=GenerationStrategy.LLM, markdown=GenerationStrategy.AUTO)

# System behavior:
# 1. Detects both fields are set
# 2. Chooses LLM strategy (higher priority)
# 3. Chooses HTML format (LLM was specified for HTML)
# 4. Generates HTML content with LLM
# 5. Populates segment.content with HTML
# 6. Still generates basic markdown for segment.markdown
# 7. Populates segment.text with OCR results
```

**Scenario 2: Legacy config accessing content field**
```python
# Old code expecting OCR text
for segment in task.segments:
    ocr_text = segment.content  # This used to work!

# What happens:
# 1. segment.content now contains generated HTML/Markdown
# 2. OCR text is in segment.text
# 3. Breaking change - old code gets wrong data
# 4. Migration required: change to segment.text
```

**Scenario 3: Mixed old/new usage**
```python
# Config uses new approach
config = GenerationConfig(format=SegmentFormat.MARKDOWN, strategy=GenerationStrategy.LLM)

# But code still accesses old fields
for segment in task.segments:
    html = segment.html      # Still populated (basic HTML)
    markdown = segment.markdown  # Contains the LLM-generated content
    content = segment.content    # Same as segment.markdown
    text = segment.text          # OCR text
```

## Benefits of New System

### Performance Improvements
- **Reduced processing time**: Generate only needed format
- **Lower resource usage**: Single format generation vs dual format
- **Faster API responses**: Less content to transfer

### Cleaner Architecture
- **Simplified configuration**: Single format choice vs multiple strategy combinations  
- **Better resource allocation**: Focus processing power on chosen format
- **Clearer content contracts**: Know exactly what format you'll receive

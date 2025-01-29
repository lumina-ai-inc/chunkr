export const curlExample = `# Start instantly with our default configurations
curl -X POST https://api.chunkr.ai/api/v1/task \\
    -H "Content-Type: multipart/form-data" \\
    -H "Authorization: YOUR_API_KEY" \\
    -F "file=@/path/to/your/file"

# Or customize the task for your use case
curl -X POST https://api.chunkr.ai/api/v1/task \\
    -H "Content-Type: multipart/form-data" \\
    -H "Authorization: YOUR_API_KEY" \\
    -F "file=@/path/to/your/file" \\
    -F 'chunk_processing={
            "ignore_headers_and_footers": false,
            "target_length": 1024
        };type=application/json' \\
    -F 'segment_processing={
            "Formula": {
                "html": "Auto",
                "markdown": "Auto"
            },
            "Table": {
                "html": "Auto",
                "markdown": "Auto"
            },
            "Picture": {
                "crop_image": "Auto"
            }
        };type=application/json' \\
    -F 'segmentation_strategy="\\"Page\\"";type=application/json'`;

// export const nodeExample = `const FormData = require('form-data');
// const fs = require('fs');
// const axios = require('axios');

// async function processDocument() {
//   const formData = new FormData();
//   formData.append('file', fs.createReadStream('/path/to/your/file'));
//   formData.append('model', 'HighQuality');
//   formData.append('target_chunk_length', '512');
//   formData.append('ocr_strategy', 'Auto');

//   try {
//     const response = await axios.post('https://api.chunkr.ai/api/v1/task', formData, {
//       headers: {
//         ...formData.getHeaders(),
//         'Authorization': 'YOUR_API_KEY'
//       }
//     });

//     console.log(response.data);
//   } catch (error) {
//     console.error('Error:', error);
//   }
// }`;

export const pythonExample = `from chunkr_ai import Chunkr
from chunkr_ai.models import (
    Configuration,
    ChunkProcessing,
    GenerationConfig,
    GenerationStrategy,
    SegmentProcessing,
    SegmentationStrategy,
    CroppingStrategy
)

chunkr = Chunkr(api_key="your_api_key")

# Start instantly with our default configurations
chunkr.upload("/path/to/your/file")

# Or customize the task for your use case
chunkr.upload("path/to/file", Configuration(
    chunk_processing=ChunkProcessing(
        ignore_headers_and_footers=False,
        target_length=1024
    ),
    segment_processing=SegmentProcessing(
        Formula=GenerationConfig(
              html=GenerationStrategy.AUTO,
              markdown=GenerationStrategy.AUTO,
        ),
        Table=GenerationConfig(
              html=GenerationStrategy.AUTO,
              markdown=GenerationStrategy.AUTO,
        ),
        Picture=GenerationConfig(
            crop_image=CroppingStrategy.AUTO
        ),
    ),
    segmentation_strategy=SegmentationStrategy.PAGE
))`;

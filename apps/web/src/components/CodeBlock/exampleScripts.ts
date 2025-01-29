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
            "target_length": 1024
        };type=application/json' \\
    -F 'segment_processing = {
        "Table": {
        "html": "LLM"
        },
        "Picture": {
        "llm": "Convert all charts to tables"
        }
    };type=application/json`;

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

chunkr = Chunkr(api_key="your_api_key")

# Start instantly with our default configurations
task = chunkr.upload("/path/to/your/file")

# Export HTML of document
task.html(output_file="output.html")

# Export markdown of document
task.markdown(output_file="output.md")

# Or customize the task for your use case (needs imports - view docs)
task = chunkr.upload("path/to/file", Configuration(
    chunk_processing=ChunkProcessing(
        target_length=1024
    ),
    segment_processing=SegmentProcessing(
        Table=GenerationConfig(
            html=GenerationStrategy.LLM,
        ),
        Picture=GenerationConfig(
            llm="Convert all charts to tables"
        ),
    ),
    # Add more configurations here
))`;

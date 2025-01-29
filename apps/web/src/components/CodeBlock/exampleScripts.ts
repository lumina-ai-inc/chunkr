export const curlExample = `curl -X POST https://api.chunkr.ai/api/v1/task \\
    -H "Content-Type: multipart/form-data" \\
    -H "Authorization: YOUR_API_KEY" \\
    -F "file=@/path/to/your/file"`;

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
task = chunkr.upload("/path/to/your/file")`;

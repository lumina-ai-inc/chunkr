export const curlExample = `curl -X POST \\
  https://api.chunkr.ai/v1/process \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "url": "https://example.com/document.pdf",
    "model": "default",
    "chunking": {
      "strategy": "semantic"
    }
  }'`;

export const nodeExample = `const axios = require('axios');

async function processDocument() {
  try {
    const response = await axios.post('https://api.chunkr.ai/v1/process', {
      url: 'https://example.com/document.pdf',
      model: 'default',
      chunking: {
        strategy: 'semantic'
      }
    }, {
      headers: {
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json'
      }
    });
    
    console.log(response.data);
  } catch (error) {
    console.error('Error:', error);
  }
}`;

export const pythonExample = `import requests

def process_document():
    url = 'https://api.chunkr.ai/v1/process'
    headers = {
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json'
    }
    payload = {
        'url': 'https://example.com/document.pdf',
        'model': 'default',
        'chunking': {
            'strategy': 'semantic'
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None`;

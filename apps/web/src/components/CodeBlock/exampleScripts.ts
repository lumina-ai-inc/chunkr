export const curlExample = `curl -X POST https://api.chunkr.ai/api/v1/task \\
    -H "Content-Type: multipart/form-data" \\
    -H "Authorization: YOUR_API_KEY" \\
    -F "file=@/path/to/your/file" \\
    -F "model=HighQuality" \\
    -F "target_chunk_length=512" \\
    -F "ocr_strategy=Auto" \\
    -F 'json_schema={
        "title": "Basket",
        "type": "object",
        "properties": [
            {
                "name": "black holes observed",
                "title": "Black Holes Observed",
                "type": "list",
                "description": "A list of black holes observed",
                "default": null
            },
            {
                "name": "implications of data",
                "title": "Implications of Data",
                "type": "string",
                "description": "The implications of the observed data",
                "default": null
            }
        ]
    };type=application/json'`;

export const nodeExample = `const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function processDocument() {
  const formData = new FormData();
  formData.append('file', fs.createReadStream('/path/to/your/file'));
  formData.append('model', 'HighQuality');
  formData.append('target_chunk_length', '512');
  formData.append('ocr_strategy', 'Auto');

  try {
    const response = await axios.post('https://api.chunkr.ai/api/v1/task', formData, {
      headers: {
        ...formData.getHeaders(),
        'Authorization': 'YOUR_API_KEY'
      }
    });
    
    console.log(response.data);
  } catch (error) {
    console.error('Error:', error);
  }
}`;

export const pythonExample = `import requests

def process_document():
    url = 'https://api.chunkr.ai/api/v1/task'
    headers = {
        'Authorization': 'YOUR_API_KEY'
    }
    
    files = {
        'file': open('/path/to/your/file', 'rb')
    }
    
    data = {
        'model': 'HighQuality',
        'target_chunk_length': '512',
        'ocr_strategy': 'Auto'
    }

    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None`;

export const rustExample = `use reqwest::blocking::multipart;
use std::fs::File;

fn process_document() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://api.chunkr.ai/api/v1/task";
    let client = reqwest::blocking::Client::new();

    let file = File::open("/path/to/your/file")?;
    let form = multipart::Form::new()
        .file("file", file)?
        .text("model", "HighQuality")
        .text("target_chunk_length", "512")
        .text("ocr_strategy", "Auto");

    let response = client
        .post(url)
        .header("Authorization", "YOUR_API_KEY")
        .multipart(form)
        .send()?;

    println!("{:#?}", response.json::<serde_json::Value>()?);
    Ok(())
}`;

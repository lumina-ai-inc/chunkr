import os
import base64
import shutil
import asyncio
import streamlit as st
from pathlib import Path
import aiohttp
import time
from typing import List, Dict, Optional, Union, Any
import dotenv
import random
dotenv.load_dotenv(override=True)

class LLMConfig:
    def __init__(self):
        self.url = os.getenv("LLM__URL")
        self.key = os.getenv("LLM__KEY")
        self.model = os.getenv("LLM__MODEL")

from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
from enum import Enum
from time import sleep

@dataclass
class ImageUrl:
    url: str

@dataclass
class ContentPart:
    content_type: str  # Using content_type instead of type as it's a Python keyword
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None
    
    def to_dict(self) -> dict:
        result = {"type": self.content_type}
        if self.text is not None:
            result["text"] = self.text
        if self.image_url is not None:
            result["image_url"] = {"url": self.image_url.url}
        return result

@dataclass
class MessageContent:
    content: Union[str, List[ContentPart]]

@dataclass
class Message:
    role: str
    content: Union[str, List[ContentPart]]
    
    def to_dict(self) -> dict:
        # Handle the flattened content field
        result = {"role": self.role}
        if isinstance(self.content, str):
            result["content"] = self.content
        else:
            result["content"] = [part.to_dict() for part in self.content]
        return result

@dataclass
class CompletionTokensDetails:
    reasoning_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None

@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    completion_tokens_details: Optional[CompletionTokensDetails] = None

@dataclass
class Choice:
    index: int
    message: Union[Message, dict]  # Allow either Message or dict
    logprobs: Optional[Any] = None
    finish_reason: str = ""

    def __post_init__(self):
        # Convert dict message to Message object if needed
        if isinstance(self.message, dict):
            self.message = Message(
                role=self.message.get('role', ''),
                content=self.message.get('content', '')
            )

@dataclass
class OpenAiResponse:
    choices: Optional[List[Choice]] = None
    created: Optional[int] = None
    id: Optional[str] = None
    model: Optional[str] = None
    object: Optional[str] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[Usage] = None
    provider: Optional[str] = None
    error: Optional[dict] = None

    def __post_init__(self):
        # Convert dict choices to Choice objects if needed
        if hasattr(self, 'choices') and self.choices is not None:
            self.choices = [
                Choice(**choice) if isinstance(choice, dict) else choice 
                for choice in self.choices
            ]
        else:
            print("No choices in response")
        
        # Print error message if present
        if self.error:
            print(f"API Error: {self.error}")

@dataclass
class OpenAiRequest:
    model: str
    messages: List[Message]
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            **({"max_tokens": self.max_completion_tokens} if self.max_completion_tokens is not None else {}),
            **({"temperature": self.temperature} if self.temperature is not None else {})
        }

def get_image_message(file_path: Path, prompt: str) -> List[Message]:
    with open(file_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()
    
    content_parts = [
        ContentPart(
            content_type="text",
            text=prompt
        ),
        ContentPart(
            content_type="image_url",
            image_url=ImageUrl(
                url=f"data:image/jpeg;base64,{base64_image}"
            )
        )
    ]
    
    return [Message(
        role="user",
        content=content_parts
    )]

async def call_llm_api(
    url: str,
    key: str,
    model: str,
    messages: List[Message],
    max_completion_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> OpenAiResponse:
    if not isinstance(url, str):
        raise ValueError(f"URL must be a string, got {type(url)}")
    if not url.startswith(('http://', 'https://')):
        raise ValueError(f"Invalid URL format: {url}")
    jitter = random.uniform(0.2, 0.8)
    sleep(jitter)
    # Simplified headers to match the working Rust implementation
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    
    request = OpenAiRequest(
        model=str(model),
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=0
    )
    if OpenAiResponse.error:
        print(f"API Error: {OpenAiResponse.error}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=request.to_dict()) as response:
                response.raise_for_status()
                data = await response.json()
                return OpenAiResponse(**data)
    except aiohttp.ClientError as e:
        print(f"API call failed: {e}")
        print(f"URL: {url}")
        print(f"Headers: {headers}")  
        raise
async def process_image(
    input_file: Path,
    output_dir: Path,
    model: str,
    prompt: str,
    llm_config: LLMConfig,
    model_shorthand: str,
    prompt_shorthand: str
) -> None:
    if not llm_config.url or not llm_config.key:
        raise ValueError("Missing LLM configuration (url or key)")

    table_name = input_file.stem
    table_dir = output_dir / table_name
    table_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(input_file, table_dir / "original.jpg")
        start_time = time.time()
        messages = get_image_message(input_file, prompt)
        
        response: OpenAiResponse = await call_llm_api(
            url=str(llm_config.url),
            key=str(llm_config.key),
            model=str(model),
            messages=messages
        )
        
        duration = time.time() - start_time
        
        # Safely extract content from the response
        if response.choices and len(response.choices) > 0:
            message_content = response.choices[0].message.content
            if isinstance(message_content, list):
                # Handle content as list of ContentPart
                content = next((part.text for part in message_content if part.text is not None), "")
            else:
                # Handle content as string
                content = str(message_content)
            
            # Save HTML output using shorthand names
            html_file = table_dir / f"{model_shorthand}_{prompt_shorthand}.html"
            with open(html_file, "w") as f:
                f.write(content)
            
            # Save metrics using shorthand names
            csv_file = table_dir / f"{model_shorthand}_{prompt_shorthand}.csv"
            csv_content = f"Model,Table,Prompt,Duration\n{model},{table_name},{prompt},{duration:.2f}\n"
            with open(csv_file, "w") as f:
                f.write(csv_content)

        else:
            print("No choices in response")
            content = "No choices in response"
            html_file = table_dir / f"{model_shorthand}_{prompt_shorthand}.html"
            with open(html_file, "w") as f:
                f.write(content)
            csv_file = table_dir / f"{model_shorthand}_{prompt_shorthand}.csv"
            csv_content = f"Model,Table,Prompt,Duration\n{model},{table_name},{prompt},{duration:.2f}\n"
            with open(csv_file, "w") as f:
                f.write(csv_content)
    except Exception as e:
        print(f"Error processing {table_name} with model {model} and prompt '{prompt}'")
        print(f"Error details: {str(e)}")
        print(f"LLM URL: {llm_config.url}")
        print(f"Input file: {input_file}")
        raise
async def main(n_files=1):
    try:
        llm_config = LLMConfig()
        
        # Validate configuration
        if not llm_config.url or not llm_config.key or not llm_config.model:
            raise ValueError("Missing required environment variables: LLM_URL, LLM_KEY, or LLM_MODEL")
            
        print(f"Using LLM URL: {llm_config.url}")  # Debug output
        
        models = {
            # "qwen2vl7b": "qwen/qwen-2-vl-7b-instruct",s
            "geminiflash8b": "google/gemini-flash-1.5-8b",
            "geminiflash": "google/gemini-flash-1.5",

            # "geminipro": "google/gemini-pro-1.5"
        }
        
        prompts = {
            # "prompt1": "Analyze this image and convert the table to HTML format maintaining the original structure. Find all text, headers, and data. DO not omit anything in the image. Try to keep the structure as close as possible while using HTML code. Make minimal and tasteful (only very sparsly or none at all) edits to handle complex tables. Keep a special eye out for Col and Row spans - accurately calculate how many you would need. Output the table directly in ```html``` tags.",
            "prompt2": "Analyze this image and convert the table to HTML format maintaining the original structure. Output the table directly in ```html``` tags."
        }
        
        input_dir = Path("input")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Get all jpg files
        image_files = list(input_dir.glob("*.jpg"))
        
        if not image_files:
            print("No .jpg files found in input directory")
            return
            
        # Process specified number of files or all if n_files is None
        files_to_process = image_files[:n_files] if n_files else image_files
        print(f"Processing {len(files_to_process)} files...")
        
        # Process selected images
        tasks = []
        for input_file in files_to_process:
            for model_shorthand, model in models.items():
                for prompt_shorthand, prompt in prompts.items():
                    tasks.append(
                        process_image(input_file, output_dir, model, prompt, llm_config, 
                                    model_shorthand, prompt_shorthand)
                    )
            
        await asyncio.gather(*tasks)
        
    except Exception as e:
        print(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys

    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(main(n))
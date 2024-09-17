from __future__ import annotations
import bentoml
from bentoml.validators import ContentType
from pathlib import Path
import subprocess
import os
import tempfile
from pydantic import Field
import base64
from typing import Dict


@bentoml.service(
    name="task",
    resources={"cpu": "2"},
    traffic={"timeout": 60}
)
class Task:
    def __init__(self) -> None:
        try:
            subprocess.run(['magick', '-version'],
                           check=True, capture_output=True)
            print("ImageMagick is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "ImageMagick is not installed or not in the system PATH")

    @bentoml.api
    def convert_to_img(
        self,
        file: Path,
        density: int = Field(default=300, description="Image density in DPI")
    ) -> Dict[int, str]:
        temp_dir = tempfile.mkdtemp()
        output_pattern = os.path.join(temp_dir, 'output-%d.png')
        result = {}
        try:
            subprocess.run(['magick', str(file), '-density', str(density), output_pattern],
                        check=True, capture_output=True, text=True)
            
            for img_file in sorted(os.listdir(temp_dir)):
                if img_file.startswith('output-') and img_file.endswith('.png'):
                    page_num = int(img_file.split('-')[1].split('.')[0])
                    with open(os.path.join(temp_dir, img_file), 'rb') as img:
                        img_base64 = base64.b64encode(img.read()).decode('utf-8')
                        result[page_num] = img_base64
            
            return result
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert image: {e.stderr}")
        finally:
            import shutil
            shutil.rmtree(temp_dir)

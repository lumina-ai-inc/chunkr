import sys
from pathlib import Path

# Add explicit debug print
print(f"Python path: {sys.path}")
print(f"Current directory: {Path.cwd()}")

from chunkr_ai.api.chunkr import Chunkr
from chunkr_ai.api.chunkr_async import ChunkrAsync 
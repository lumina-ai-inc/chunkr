from typing import Optional, runtime_checkable, Protocol
from httpx import AsyncClient


@runtime_checkable
class ChunkrClientProtocol(Protocol):
    """Protocol defining the interface for Chunkr clients"""
    
    raise_on_failure: bool = True
    _client: Optional[AsyncClient] = None

    def _headers(self) -> dict:
        """Return headers required for API requests"""
        ...

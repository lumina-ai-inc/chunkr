from typing import Optional, runtime_checkable, Protocol
from requests import Session
from httpx import AsyncClient

@runtime_checkable
class ChunkrClientProtocol(Protocol):
    """Protocol defining the interface for Chunkr clients"""
    url: str
    _api_key: str
    _session: Optional[Session] = None
    _client: Optional[AsyncClient] = None

    def get_api_key(self) -> str:
        """Get the API key"""
        ...

    def _headers(self) -> dict:
        """Return headers required for API requests"""
        ...
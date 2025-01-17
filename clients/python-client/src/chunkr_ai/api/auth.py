class HeadersMixin:
    """Mixin class for handling authorization headers"""

    def get_api_key(self) -> str:
        """Get the API key"""
        if not hasattr(self, "_api_key") or not self._api_key:
            raise ValueError("API key not set")
        return self._api_key

    def _headers(self) -> dict:
        """Generate authorization headers"""
        return {"Authorization": self.get_api_key()}

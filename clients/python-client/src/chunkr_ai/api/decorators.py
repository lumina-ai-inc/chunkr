import functools
import asyncio
import httpx
from typing import Callable, Any, TypeVar, Awaitable, Union, overload
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

T = TypeVar('T')
P = ParamSpec('P')

_sync_loop = None

@overload
def anywhere() -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Union[Awaitable[T], T]]]: ...

def anywhere():
    """Decorator that allows an async function to run anywhere - sync or async context."""
    def decorator(async_func: Callable[P, Awaitable[T]]) -> Callable[P, Union[Awaitable[T], T]]:
        @functools.wraps(async_func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[Awaitable[T], T]:
            global _sync_loop
            try:
                asyncio.get_running_loop()
                return async_func(*args, **kwargs) 
            except RuntimeError:
                if _sync_loop is None:
                    _sync_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_sync_loop) 
                try:
                    return _sync_loop.run_until_complete(async_func(*args, **kwargs))
                finally:
                    asyncio.set_event_loop(None)
        return wrapper
    return decorator

def ensure_client() -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator that ensures a valid httpx.AsyncClient exists before executing the method"""
    def decorator(async_func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(async_func)
        async def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
            if not self._client or self._client.is_closed:
                self._client = httpx.AsyncClient()
            return await async_func(self, *args, **kwargs)
        return wrapper
    return decorator

def require_task() -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator that ensures task has required attributes and valid client before execution"""
    def decorator(async_func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(async_func)
        async def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
            if not self.task_url:
                raise ValueError("Task URL not found")
            if not self._client:
                raise ValueError("Client not found")
            if not self._client._client or self._client._client.is_closed:
                self._client._client = httpx.AsyncClient()
            return await async_func(self, *args, **kwargs)
        return wrapper
    return decorator
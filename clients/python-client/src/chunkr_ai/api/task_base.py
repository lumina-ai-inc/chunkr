from abc import ABC, abstractmethod
from .config import Configuration

class TaskBase(ABC):
    @abstractmethod
    def poll(self):
        pass

    @abstractmethod
    def update(self, config: Configuration):
        pass

    @abstractmethod
    def cancel(self):
        pass

    @abstractmethod
    def delete(self):
        pass

    @abstractmethod
    def html(self) -> str:
        pass

    @abstractmethod
    def markdown(self) -> str:
        pass

    @abstractmethod
    def content(self) -> str:
        pass

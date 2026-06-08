from abc import ABC, abstractmethod
from enrichment.metadata_schema import ParsedDocument


class BaseProcessor(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        ...

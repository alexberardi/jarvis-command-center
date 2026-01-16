from typing import Protocol, Dict

class IJarvisContextProvider(Protocol):
    key: str # key for the cache dictionary
    def get_context(self, user_text: str, context_map: Dict[str, str]) -> str:
        ...


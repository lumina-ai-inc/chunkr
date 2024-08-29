from enum import Enum


class TokenType(Enum):
    FORMULA = "Formula"
    FOOTNOTE = "Footnote"
    LIST_ITEM = "List item"
    TABLE = "Table"
    PICTURE = "Picture"
    TITLE = "Title"
    TEXT = "Text"
    PAGE_HEADER = "Page header"
    SECTION_HEADER = "Section header"
    CAPTION = "Caption"
    PAGE_FOOTER = "Page footer"

    @staticmethod
    def from_text(text: str):
        try:
            return TokenType[text.upper()]
        except KeyError:
            return TokenType.TEXT

    @staticmethod
    def from_index(index: int):
        try:
            return list(TokenType)[index]
        except IndexError:
            return TokenType.TEXT.name.lower()

    @staticmethod
    def from_value(value: str):
        for token_type in TokenType:
            if token_type.value == value:
                return token_type
        return TokenType.TEXT

    def get_index(self) -> int:
        return list(TokenType).index(self)

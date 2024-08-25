from enum import Enum


class TaskMistakesType(Enum):
    CORRECT = "CORRECT"
    WRONG = "WRONG"
    MISSING = "MISSING"

    @staticmethod
    def contains(key: str):
        return key.upper() in [e.value for e in TaskMistakesType]

    @staticmethod
    def from_text(text: str):
        try:
            return TaskMistakesType[text.upper()]
        except KeyError:
            return TaskMistakesType.WRONG

    def get_index(self) -> int:
        return list(TaskMistakesType).index(self)

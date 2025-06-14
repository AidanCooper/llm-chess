from enum import Enum


class MoveNotation(Enum):
    UCI = "UCI"
    SAN = "SAN"


class APIResponseFormat(Enum):
    ENUM = "enum"
    STRUCTURED = "structured"
    JSON = "json"
    TEXT = "text"
    MULTI_TURN = "multi_turn"

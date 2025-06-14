from abc import ABC, abstractmethod

import chess

from llm_chess.core.enums import APIResponseFormat, MoveNotation
from llm_chess.utils.format import convert_move_to_str

ENUM_RESPONSE = "Using {notation} notation, respond directly with the best move."

JSON_RESPONSE = """
Using {notation} notation, provide the best legal move as a JSON object with the key "move".
Example JSON output:
{{
    "move": "{notated_move}"
}}
"""


class ResponseInstructionsMixin:
    api_response_format: APIResponseFormat
    move_notation: MoveNotation

    def response_instructions(self, example_move: chess.Move | None = None) -> str:
        """
        Generate instructions for the model on how to respond.

        Args:
            example_move: A chess.Move object to use as an example. If None, this
                defaults to "e2e4".

        Returns:
            A string with instructions for the model
        """
        if example_move is None:
            example_move = chess.Move.from_uci("e2e4")

        if self.api_response_format in (
            APIResponseFormat.TEXT,
            APIResponseFormat.ENUM,
            APIResponseFormat.MULTI_TURN,
        ):
            return ENUM_RESPONSE.format(notation=self.move_notation.value)
        elif self.api_response_format in (APIResponseFormat.JSON, APIResponseFormat.STRUCTURED):
            return JSON_RESPONSE.format(
                notation=self.move_notation.value,
                notated_move=convert_move_to_str(chess.Board(), example_move, self.move_notation),
            )
        else:
            raise ValueError(f"Unknown API response format: {self.api_response_format}")


class PromptConfig(ABC):
    """Abstract base class for chess prompts."""

    def __init__(
        self,
        move_notation: MoveNotation = MoveNotation.UCI,
        api_response_format: APIResponseFormat = APIResponseFormat.TEXT,
        move_response_has_leading_space: bool = False,
    ):
        """
        Args:
            move_notation: The notation format expected for moves in the response
            api_response_format: Format specification for the API response
            move_response_has_leading_space: Whether to include a leading space in the
                LLM's move responses. This may work in conjunction with the prompt
                itself, by including or omitting a corresponding trailing space. This is
                useful, as some models are extremely sensitive to move formatting.
        """
        self.move_notation = move_notation
        self.api_response_format = api_response_format
        self.move_response_has_leading_space = move_response_has_leading_space

    @abstractmethod
    def build_prompt(self, board: chess.Board) -> str:
        """
        Build a prompt string for the given board state.

        Args:
            board: The chess board to build a prompt for

        Returns:
            A formatted prompt string
        """
        pass

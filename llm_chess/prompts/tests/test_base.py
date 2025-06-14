import chess
import pytest

from llm_chess.core.enums import APIResponseFormat, MoveNotation
from llm_chess.prompts.base import PromptConfig, ResponseInstructionsMixin


class DummyPromptConfig(PromptConfig, ResponseInstructionsMixin):
    """Concrete implementation of PromptConfig for testing."""

    def build_prompt(self, board: chess.Board) -> str:
        return "Test prompt for board."


def test_response_instructions_text() -> None:
    config = DummyPromptConfig(
        move_notation=MoveNotation.UCI, api_response_format=APIResponseFormat.TEXT
    )
    instructions = config.response_instructions()
    assert "Using UCI notation, respond directly with the best move." in instructions


def test_response_instructions_json() -> None:
    config = DummyPromptConfig(
        move_notation=MoveNotation.UCI,
        api_response_format=APIResponseFormat.JSON,
    )
    instructions = config.response_instructions()
    assert "Using UCI notation, provide the best legal move as a JSON object" in instructions
    assert '"move": "e2e4"' in instructions


def test_response_instructions_invalid_format() -> None:
    with pytest.raises(ValueError, match="Unknown API response format"):
        DummyPromptConfig(
            move_notation=MoveNotation.UCI, api_response_format="INVALID_FORMAT"  # type: ignore
        ).response_instructions()


def test_build_prompt() -> None:
    board = chess.Board()
    config = DummyPromptConfig()
    prompt = config.build_prompt(board)
    assert prompt == "Test prompt for board."

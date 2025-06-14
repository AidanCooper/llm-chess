import chess
import pytest

from llm_chess.conftest import MockPromptConfig
from llm_chess.players.llm.base import LLMPlayer
from llm_chess.prompts.base import PromptConfig


class MockLLMPlayer(LLMPlayer):
    """Mock implementation of MockLLMPlayer for testing."""

    def __init__(
        self,
        mock_response: str = "e2e4",
        name: str = "Mock",
        prompt_config: PromptConfig | None = None,
    ):
        prompt_config = prompt_config or MockPromptConfig()
        super().__init__(name, prompt_config)
        self.mock_response = mock_response

    def _get_model_response(self, board: chess.Board) -> str:
        return self.mock_response


@pytest.fixture
def mock_llm_player(mock_prompt_config: MockPromptConfig) -> MockLLMPlayer:
    return MockLLMPlayer(prompt_config=mock_prompt_config)


def test_valid_move_from_starting_position(
    starting_board: chess.Board, mock_llm_player: MockLLMPlayer
) -> None:
    move = mock_llm_player._get_move(starting_board)
    assert move == chess.Move.from_uci("e2e4")
    assert move in starting_board.legal_moves

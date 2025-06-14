import chess
import pytest
from dotenv import load_dotenv

from llm_chess.core.enums import APIResponseFormat, MoveNotation
from llm_chess.core.player import ChessPlayer
from llm_chess.prompts.base import PromptConfig

load_dotenv()


@pytest.fixture
def starting_board() -> chess.Board:
    return chess.Board()


@pytest.fixture
def starting_board_black(starting_board: chess.Board) -> chess.Board:
    starting_board.push_san("d4")
    return starting_board


@pytest.fixture
def starting_board_str_without_symbols() -> str:
    return """
8 r n b q k b n r
7 p p p p p p p p
6 . . . . . . . .
5 . . . . . . . .
4 . . . . . . . .
3 . . . . . . . .
2 P P P P P P P P
1 R N B Q K B N R
  a b c d e f g h
""".strip()


@pytest.fixture
def starting_board_str() -> str:
    return """
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
6 . . . . . . . .
5 . . . . . . . .
4 . . . . . . . .
3 . . . . . . . .
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
  a b c d e f g h
""".strip()


@pytest.fixture
def starting_board_str_black_without_symbols() -> str:
    return """
1 R N B Q K B N R
2 P P P . P P P P
3 . . . . . . . .
4 . . . P . . . .
5 . . . . . . . .
6 . . . . . . . .
7 p p p p p p p p
8 r n b q k b n r
  a b c d e f g h
""".strip()


@pytest.fixture
def starting_board_str_black() -> str:
    return """
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
2 ♙ ♙ ♙ . ♙ ♙ ♙ ♙
3 . . . . . . . .
4 . . . ♙ . . . .
5 . . . . . . . .
6 . . . . . . . .
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
  a b c d e f g h
""".strip()


@pytest.fixture
def legal_starting_moves(starting_board: chess.Board) -> list[chess.Move]:
    return list(starting_board.legal_moves)


@pytest.fixture
def legal_starting_moves_formatted_uci(
    legal_starting_moves: list[chess.Move], starting_board: chess.Board
) -> str:
    return ", ".join(starting_board.uci(move) for move in legal_starting_moves)


@pytest.fixture
def legal_starting_moves_formatted_san(
    legal_starting_moves: list[chess.Move], starting_board: chess.Board
) -> str:
    return ", ".join(starting_board.san(move) for move in legal_starting_moves)


@pytest.fixture
def legal_starting_moves_black(starting_board_black: chess.Board) -> list[chess.Move]:
    return list(starting_board_black.legal_moves)


@pytest.fixture
def legal_starting_moves_formatted_black_uci(
    legal_starting_moves_black: list[chess.Move], starting_board_black: chess.Board
) -> str:
    return ", ".join(starting_board_black.uci(move) for move in legal_starting_moves_black)


@pytest.fixture
def legal_starting_moves_formatted_black_san(
    legal_starting_moves_black: list[chess.Move], starting_board_black: chess.Board
) -> str:
    return ", ".join(starting_board_black.san(move) for move in legal_starting_moves_black)


@pytest.fixture
def stalemate_board() -> chess.Board:
    return chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")


class MockPromptConfig(PromptConfig):
    """Mock implementation of PromptConfig for testing."""

    def __init__(
        self,
        move_notation: MoveNotation = MoveNotation.UCI,
        api_response_format: APIResponseFormat = APIResponseFormat.STRUCTURED,
    ):
        super().__init__(move_notation, api_response_format)
        self.template = "Mock prompt"

    def build_prompt(self, board: chess.Board) -> str:
        return self.template


@pytest.fixture
def mock_prompt_config() -> MockPromptConfig:
    return MockPromptConfig()


class MockChessPlayer(ChessPlayer):
    """Mock implementation of ChessPlayer for testing."""

    def __init__(self, mock_move: str = "e2e4", name: str = "Mock"):
        super().__init__(name)
        self.mock_move = mock_move

    def _get_move(self, board: chess.Board) -> chess.Move:
        return chess.Move.from_uci(self.mock_move)

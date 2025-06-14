import chess
import pytest

from llm_chess.players.random import RandomPlayer


@pytest.fixture
def random_player() -> RandomPlayer:
    """Fixture for initializing a HumanPlayer instance."""
    return RandomPlayer("Random")


def test_random_player_selects_legal_move(
    random_player: RandomPlayer,
    starting_board: chess.Board,
    legal_starting_moves: list[chess.Move],
) -> None:
    move = random_player._get_move(starting_board)
    assert move in legal_starting_moves

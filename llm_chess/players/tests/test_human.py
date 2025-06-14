from unittest.mock import patch

import pytest
from chess import Board, Move  # Assuming these are the correct imports for Board and Move

from llm_chess.players.human import HumanPlayer


@pytest.fixture
def human_player() -> HumanPlayer:
    """Fixture for initializing a HumanPlayer instance."""
    return HumanPlayer("Human")


def test_human_player_accepts_valid_move(
    human_player: HumanPlayer, starting_board: Board, legal_starting_moves: list[Move]
) -> None:
    """Test that HumanPlayer accepts a valid move."""
    valid_move = legal_starting_moves[0].uci()
    with patch("builtins.input", return_value=valid_move):
        selected_move = human_player._get_move(starting_board)
    assert selected_move.uci() == valid_move, "Valid move was not accepted."


def test_human_player_rejects_invalid_move(
    human_player: HumanPlayer, starting_board: Board, legal_starting_moves: list[Move]
) -> None:
    """Test that HumanPlayer rejects an invalid move and prompts again."""
    invalid_move = "a1a1"
    valid_move = legal_starting_moves[0].uci()
    inputs = [invalid_move, valid_move]

    with patch("builtins.input", side_effect=inputs):
        selected_move = human_player._get_move(starting_board)
    assert selected_move.uci() == valid_move, "Valid move was not accepted after invalid input."


def test_human_player_handles_invalid_format(
    human_player: HumanPlayer, starting_board: Board, legal_starting_moves: list[Move]
) -> None:
    """Test that HumanPlayer handles input with an invalid format gracefully."""
    invalid_format = "invalid"
    valid_move = legal_starting_moves[0].uci()
    inputs = [invalid_format, valid_move]

    with patch("builtins.input", side_effect=inputs):
        selected_move = human_player._get_move(starting_board)
    assert (
        selected_move.uci() == valid_move
    ), "Valid move was not accepted after invalid input format."

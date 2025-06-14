import chess
import pytest

from llm_chess.conftest import MockChessPlayer


def test_valid_move_from_starting_position(starting_board: chess.Board) -> None:
    player = MockChessPlayer("e2e4")
    move = player.make_move(starting_board)
    assert move == chess.Move.from_uci("e2e4")
    assert move in starting_board.legal_moves


def test_no_legal_moves_returns_none(stalemate_board: chess.Board) -> None:
    player = MockChessPlayer("e2e4")
    move = player.make_move(stalemate_board)
    assert move is None


@pytest.mark.parametrize(
    "invalid_response",
    [
        "invalid",
        "xx99",
        "e2e9",
        "",
        "e2",
    ],
)
def test_invalid_move_format_raises_value_error(
    starting_board: chess.Board, invalid_response: str
) -> None:
    player = MockChessPlayer(invalid_response)
    with pytest.raises(ValueError):
        player.make_move(starting_board)


def test_illegal_move_raises_value_error(starting_board: chess.Board) -> None:
    # e2e5 is valid UCI format but not a legal chess move
    player = MockChessPlayer("e2e5")
    with pytest.raises(ValueError):
        player.make_move(starting_board)

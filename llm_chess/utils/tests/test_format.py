import chess
import pytest

from llm_chess.core.enums import MoveNotation
from llm_chess.utils.format import (
    convert_move_to_str,
    convert_str_to_move,
    format_legal_moves,
    format_moves_history,
)


def test_convert_str_to_move(starting_board: chess.Board) -> None:
    move_uci = "e2e4"
    move_san = "e4"
    assert convert_str_to_move(starting_board, move_uci, MoveNotation.UCI) == chess.Move.from_uci(
        move_uci
    )
    assert convert_str_to_move(starting_board, move_san, MoveNotation.SAN) == chess.Move.from_uci(
        move_uci
    )

    with pytest.raises(ValueError):
        convert_str_to_move(starting_board, "invalid", MoveNotation.UCI)


def test_convert_move_to_str(starting_board: chess.Board) -> None:
    move = chess.Move.from_uci("e2e4")
    assert convert_move_to_str(starting_board, move, MoveNotation.UCI) == "e2e4"
    assert convert_move_to_str(starting_board, move, MoveNotation.SAN) == "e4"

    with pytest.raises(ValueError):
        convert_move_to_str(starting_board, move, "INVALID_NOTATION")  # type: ignore


def test_format_legal_moves(starting_board: chess.Board) -> None:
    uci_moves = format_legal_moves(starting_board, MoveNotation.UCI)
    san_moves = format_legal_moves(starting_board, MoveNotation.SAN)

    assert isinstance(uci_moves, list)
    assert isinstance(san_moves, list)
    assert all(isinstance(move, str) for move in uci_moves)
    assert all(isinstance(move, str) for move in san_moves)

    with pytest.raises(ValueError):
        format_legal_moves(starting_board, "INVALID_NOTATION")  # type: ignore


@pytest.mark.parametrize(
    "move_notation,expected",
    [
        (MoveNotation.UCI, ["e2e4", "e7e5", "g1f3"]),
        (MoveNotation.SAN, ["e4", "e5", "Nf3"]),
    ],
)
def test_format_moves_history(
    starting_board: chess.Board, move_notation: MoveNotation, expected: list[str]
) -> None:
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
    ]
    for move in moves:
        starting_board.push(move)

    history = format_moves_history(starting_board, move_notation)

    assert history == expected

    with pytest.raises(ValueError):
        format_moves_history(starting_board, "INVALID_NOTATION")  # type: ignore

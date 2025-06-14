from pathlib import Path

import chess
import pytest

from llm_chess.utils.parse import parse_board_from_pgn_file

ROOT_DIR = Path(__file__).resolve().parents[3]
PGN_PATH = ROOT_DIR / "game_logs" / "test.pgn"
BAD_PATH = ROOT_DIR / "game_logs" / "bad.pgn"


def test_parse_valid_pgn() -> None:
    board = parse_board_from_pgn_file(PGN_PATH)
    assert isinstance(board, chess.Board)
    assert board.is_valid()
    assert board.fullmove_number > 0


def test_parse_final_position() -> None:
    board = parse_board_from_pgn_file(PGN_PATH)
    # Final position: White King f1, Black King e8, no other pieces
    assert board.king(chess.WHITE) == chess.H7
    assert board.king(chess.BLACK) == chess.E8
    assert board.piece_at(chess.A1) is None
    assert board.is_valid()


def test_bad_pgn_path() -> None:
    with pytest.raises(FileNotFoundError):
        parse_board_from_pgn_file(BAD_PATH)

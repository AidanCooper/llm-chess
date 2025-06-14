from pathlib import Path

import chess
import chess.pgn


def parse_board_from_pgn_file(pgn_file_path: str | Path) -> chess.Board:
    """
    Parses the final board position from the first game in a PGN file.

    Args:
        pgn_file_path (str): Path to the PGN file.

    Returns:
        chess.Board: Final board state after all mainline moves.

    Raises:
        ValueError: If no valid game is found in the PGN file.
    """
    with open(pgn_file_path, encoding="utf-8") as pgn:
        game = chess.pgn.read_game(pgn)
        if game is None:
            raise ValueError(f"No valid game found in PGN file: {pgn_file_path}")

        board = game.board()
        for move in game.mainline_moves():
            board.push(move)

        return board

from datetime import date
from pathlib import Path

import chess


def write_board_to_pgn_file(
    board: chess.Board,
    write_dir: Path,
    file_name: str = "game.pgn",
    white_name: str = "White Player",
    black_name: str = "Black Player",
    result: str | None = None,
) -> None:
    if result is None:
        result = board.result()

    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": "LLM‑Calibration",
            "Site": "Local",
            "Date": date.today().isoformat(),
            "White": white_name,
            "Black": black_name,
            "Result": result,
        }
    )
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)

    pgn_path = write_dir / f"{file_name}"
    with open(pgn_path, "w") as f:
        print(game, file=f)

import random

import chess

from llm_chess.core.player import ChessPlayer


class RandomPlayer(ChessPlayer):
    """Random player interface."""

    def _get_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)

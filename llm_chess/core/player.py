from abc import ABC, abstractmethod

import chess


class ChessPlayer(ABC):
    """Abstract base class for chess players."""

    def __init__(self, name: str):
        self.name = name

    def make_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        move = self._get_move(board)
        if move not in legal_moves:
            raise ValueError(f"Invalid move: {move}")
        return move

    @abstractmethod
    def _get_move(self, board: chess.Board) -> chess.Move:
        pass

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}'"

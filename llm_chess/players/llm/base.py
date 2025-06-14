from abc import ABC, abstractmethod

import chess

from llm_chess.core.player import ChessPlayer
from llm_chess.prompts.base import PromptConfig
from llm_chess.utils.format import convert_str_to_move


class LLMPlayer(ChessPlayer, ABC):
    """Abstract base class for LLM-based players."""

    def __init__(self, name: str, prompt_config: PromptConfig):
        super().__init__(name)
        self.prompt_config = prompt_config

    def _get_move(self, board: chess.Board) -> chess.Move:
        notation = self.prompt_config.move_notation
        move_str = self._get_model_response(board)
        return convert_str_to_move(board, move_str, notation)

    @abstractmethod
    def _get_model_response(
        self,
        board: chess.Board,
    ) -> str:
        pass

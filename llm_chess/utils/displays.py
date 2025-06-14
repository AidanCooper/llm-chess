from abc import ABC, abstractmethod

import chess

from llm_chess.prompts.base import PromptConfig
from llm_chess.prompts.text_board import board_to_text


class BoardDisplayer(ABC):
    @abstractmethod
    def display(self, board: chess.Board, ended: bool = False) -> str:
        pass


class TextDisplayer(BoardDisplayer):
    def display(self, board: chess.Board, ended: bool = False) -> str:
        return board_to_text(board)


class PromptDisplayer(BoardDisplayer):
    def __init__(self, prompt_config: PromptConfig):
        self.prompt_config = prompt_config

    def display(self, board: chess.Board, ended: bool = False) -> str:
        return self.prompt_config.build_prompt(board)

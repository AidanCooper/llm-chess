import chess

from llm_chess.core.enums import APIResponseFormat, MoveNotation
from llm_chess.prompts.base import PromptConfig, ResponseInstructionsMixin
from llm_chess.utils.format import format_legal_moves

PROMPT_TEMPLATE = """
You are playing as {colour}. The current board state in FEN format is as follows:
{fen_board}

Your legal moves are:
{formatted_legal_moves}

{response_instructions}
""".strip()


class FENPromptConfig(PromptConfig, ResponseInstructionsMixin):
    """
    Config for generating a FEN representation of a chess board.
    """

    def __init__(
        self,
        *,
        move_notation: MoveNotation = MoveNotation.UCI,
        api_response_format: APIResponseFormat = APIResponseFormat.TEXT,
        include_response_instructions: bool = True,
    ):
        """
        Args:
            move_notation: The notation format expected for moves in the response
            api_response_format: Format specification for the API response
            include_response_instructions: Whether to include instructions for the
                model on how to respond
        """
        super().__init__(move_notation, api_response_format)
        self.template = PROMPT_TEMPLATE
        self.include_response_instructions = include_response_instructions

    def build_prompt(self, board: chess.Board) -> str:
        colour = "white" if board.turn == chess.WHITE else "black"
        formatted_legal_moves = format_legal_moves(board, self.move_notation)
        response_instructions = (
            self.response_instructions() if self.include_response_instructions else ""
        )
        return self.template.format(
            colour=colour,
            fen_board=board.fen(),
            formatted_legal_moves=", ".join(formatted_legal_moves),
            response_instructions=response_instructions,
        )

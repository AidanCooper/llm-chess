import chess

from llm_chess.core.enums import APIResponseFormat, MoveNotation
from llm_chess.prompts.base import PromptConfig, ResponseInstructionsMixin

PROMPT_TEMPLATE = """
You are a chess engine, playing as {colour}. {response_instructions} {make_move}
""".strip()


class MultiTurnPromptConfig(PromptConfig, ResponseInstructionsMixin):
    """
    Config for generating a multi-turn prompt representation of a chess game.
    """

    def __init__(
        self,
        *,
        move_notation: MoveNotation = MoveNotation.SAN,
        api_response_format: APIResponseFormat = APIResponseFormat.MULTI_TURN,
        include_response_instructions: bool = True,
    ):
        """
        Args:
            move_notation: The notation format expected for moves in the response.
            api_response_format: Format specification for the API response. This should
                always be MULTI_TURN for this prompt config.
            include_response_instructions: Whether to include instructions for the
                model on how to respond.
        """
        if api_response_format != APIResponseFormat.MULTI_TURN:
            raise ValueError(
                "Multi-turn prompt only supports multi_turn response format, "
                f"but got {api_response_format}"
            )

        super().__init__(move_notation, api_response_format)
        self.template = PROMPT_TEMPLATE
        self.include_response_instructions = include_response_instructions

    def build_prompt(self, board: chess.Board) -> str:
        """
        Build a prompt for use in a multi-turn conversation with an LLM.

        Args:
            board: The current chess board state

        Returns:
            A string representing the prompt
        """
        colour = "white" if board.turn == chess.WHITE else "black"
        response_instructions = (
            self.response_instructions() if self.include_response_instructions else ""
        )
        make_move = "Make your first move." if colour == "white" else ""
        return self.template.format(
            colour=colour, response_instructions=response_instructions, make_move=make_move
        )

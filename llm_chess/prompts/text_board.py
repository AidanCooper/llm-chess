import chess

from llm_chess.core.enums import APIResponseFormat, MoveNotation
from llm_chess.prompts.base import PromptConfig, ResponseInstructionsMixin
from llm_chess.utils.format import format_legal_moves

PROMPT_TEMPLATE = """
It is your turn as {colour}. The current chess position is as follows:
{board_str}
{piece_explanation}
Your legal moves are:
{formatted_legal_moves}

{response_instructions}
""".strip()

PIECE_EXPLANATION = """
Pieces: R = rook, N = knight, B = bishop, Q = queen, K = king, P = pawn.
Upper case letters denote white pieces, and lower case letters denote black pieces.
"""


def board_to_text(board: chess.Board, flip_board: bool = False, piece_symbols: bool = True) -> str:
    piece_symbols_map = {
        "K": "♔",
        "Q": "♕",
        "R": "♖",
        "B": "♗",
        "N": "♘",
        "P": "♙",
        "k": "♚",
        "q": "♛",
        "r": "♜",
        "b": "♝",
        "n": "♞",
        "p": "♟",
    }
    translation_table = str.maketrans(piece_symbols_map)
    str_board = str(board).translate(translation_table) if piece_symbols else str(board)

    rows_with_labels = [f"{8 - i} {line}" for i, line in enumerate(str_board.splitlines())]
    if flip_board:
        rows_with_labels = rows_with_labels[::-1]
    str_board = "\n".join(rows_with_labels) + "\n  a b c d e f g h"

    return str(str_board)


class TextBoardPromptConfig(PromptConfig, ResponseInstructionsMixin):
    """
    Config for generating a 2D textual representation of a chess board.
    """

    def __init__(
        self,
        *,
        move_notation: MoveNotation = MoveNotation.UCI,
        api_response_format: APIResponseFormat = APIResponseFormat.TEXT,
        include_response_instructions: bool = True,
        piece_symbols: bool = True,
        flip_board: bool = True,
    ):
        """
        Args:
            move_notation: The notation format expected for moves in the response
            api_response_format: Format specification for the API response
            include_response_instructions: Whether to include instructions for the
                model on how to respond
            piece_symbols: Whether to use piece symbols in the board representation. If
                False, use letters (R, N, B, Q, K, P) instead.
            flip_board: Whether to flip the board for black's turns
        """
        super().__init__(move_notation, api_response_format)
        self.template = PROMPT_TEMPLATE
        self.include_response_instructions = include_response_instructions
        self.piece_symbols = piece_symbols
        self.flip_board = flip_board

    def build_prompt(self, board: chess.Board) -> str:
        colour = "white" if board.turn == chess.WHITE else "black"
        flip_board = colour == "black" if self.flip_board else False

        board_str = board_to_text(board, flip_board=flip_board, piece_symbols=self.piece_symbols)
        piece_explanation = PIECE_EXPLANATION if not self.piece_symbols else ""
        formatted_legal_moves = format_legal_moves(board, self.move_notation)
        response_instructions = (
            self.response_instructions() if self.include_response_instructions else ""
        )

        return self.template.format(
            colour=colour,
            board_str=board_str,
            piece_explanation=piece_explanation,
            formatted_legal_moves=", ".join(formatted_legal_moves),
            response_instructions=response_instructions,
        )

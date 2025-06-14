import chess

from llm_chess.core.player import ChessPlayer
from llm_chess.prompts.text_board import board_to_text


class HumanPlayer(ChessPlayer):
    """Human player interface."""

    def _get_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        print("\033[H\033[J")
        print("\nCurrent board position:")
        print(board_to_text(board))
        print("\nLegal moves:", ", ".join(move.uci() for move in legal_moves))

        while True:
            try:
                move_uci = input("Enter your move in UCI notation (e.g., 'e2e4'): ")
                move = chess.Move.from_uci(move_uci)
                if move in legal_moves:
                    return move
                print("Invalid move. Please try again.")
            except ValueError:
                print("Invalid input format. Please use UCI notation.")

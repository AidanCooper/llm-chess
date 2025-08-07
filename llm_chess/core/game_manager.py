import random
import sys
import time

import chess

from llm_chess.core.player import ChessPlayer
from llm_chess.utils.displays import BoardDisplayer

# Attempt to import display from IPython. Only available in IPython environments.
try:
    from IPython.display import clear_output

    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False


class GameManager:
    def _print_board(
        self,
        board: chess.Board,
        displayer: BoardDisplayer,
        ended: bool = False,
        sleep_time: float = 0.1,
    ) -> None:
        status = "Final" if ended else "Current"

        # Clear the output based on the environment
        if _IPYTHON_AVAILABLE and "ipykernel" in sys.modules:
            # Running in a Jupyter notebook or IPython console
            clear_output(wait=True)
        else:
            # Running in a standard console
            print("\033[H\033[J")  # ANSI escape code to clear console

        print(f"\n{status} board position:")
        print(displayer.display(board, ended))

        time.sleep(sleep_time)

    def play_game(
        self,
        white: ChessPlayer,
        black: ChessPlayer,
        board: chess.Board | None,
        displayer: BoardDisplayer | None = None,
        print_move: bool = False,
        sleep_time: float = 0.1,
        max_half_moves: int = 400,
        n_randomised_starting_half_moves: int = 0,
    ) -> tuple[chess.Board, str]:
        """
        Plays a game of chess between two players.

        Args:
            white (ChessPlayer): The player playing as white.
            black (ChessPlayer): The player playing as black.
            board (chess.Board | None): The initial board state. If None, a new board
                is created.
            displayer (BoardDisplayer | None): Optional displayer for the board.
            sleep_time (float): Time to wait between moves for display purposes.
            max_half_moves (int): Maximum number of half-moves allowed in the game.
            n_randomised_starting_half_moves (int): Number of random moves to make
                before before using the players' strategies.

        Returns:
            tuple[chess.Board, str]: The final board state and the game result.

        Raises:
            ValueError: If the board is not valid or if a player makes an illegal move.
        """
        if board is None:
            board = chess.Board()

        n_half_moves = 0
        while not board.is_game_over() and n_half_moves <= max_half_moves:
            if displayer:
                self._print_board(board, displayer, sleep_time=sleep_time)

            current_player = white if board.turn == chess.WHITE else black
            move = None
            if n_half_moves < n_randomised_starting_half_moves:
                move = random.choice(list(board.legal_moves))
            else:
                try:
                    move = current_player.make_move(board)
                except Exception as e:
                    print(f"Error: {e}")
                    return board, f"Illegal move by {current_player.name}"

            if move is None:
                break

            if print_move:
                length = max(len(white.name), len(black.name))
                print(f"    {current_player.name:<{length}} plays: {board.san(move)}")

            board.push(move)
            n_half_moves += 1

        if displayer:
            self._print_board(board, displayer, ended=True)

        return board, board.result()

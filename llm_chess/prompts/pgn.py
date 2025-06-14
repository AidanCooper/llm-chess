import re

import chess
import chess.pgn

from llm_chess.core.enums import APIResponseFormat, MoveNotation
from llm_chess.prompts.base import PromptConfig


class PGNPromptConfig(PromptConfig):
    """
    Config for generating a PGN representation of a chess game.
    """

    def __init__(
        self,
        *,
        move_notation: MoveNotation = MoveNotation.SAN,
        api_response_format: APIResponseFormat = APIResponseFormat.ENUM,
        move_response_has_leading_space: bool = True,
        event: str = "World Championship",
        site: str = "Moscow",
        date: str = "2025.01.01",
        round: str = "1",
        white_player: str = "Kasparov, Garry",
        black_player: str = "Carlsen, Magnus",
        result: str | None = None,
        white_player_elo: int = 2800,
        black_player_elo: int = 2800,
    ):
        """
        Args:
            move_notation: The notation format expected for moves in the response. For
                PGN, this is always SAN.
            api_response_format: Format specification for the API response
            move_response_has_leading_space: Whether to include a leading space in the
                LLM's move responses.
            event: Name of the event
            site: Location of the event
            date: Date of the event in YYYY.MM.DD format
            white_player: Name of the white player
            black_player: Name of the black player
            result: Result of the game (e.g., "1-0", "0-1", "1/2-1/2"). If None, the
                result is generated dynamically to indicate a win for the player whose
                turn it is.
            white_player_elo: Elo rating of the white player
            black_player_elo: Elo rating of the black player
        """
        if move_notation != MoveNotation.SAN:
            raise ValueError(f"PGN format only supports SAN notation, but got {move_notation}")

        super().__init__(move_notation, api_response_format, move_response_has_leading_space)
        self.event = event
        self.site = site
        self.date = date
        self.round = round
        self.white_player = white_player
        self.black_player = black_player
        self.result = result
        self.white_player_elo = white_player_elo
        self.black_player_elo = black_player_elo

    def _get_result(self, is_white_turn: bool) -> str:
        if self.result is not None:
            return self.result
        return "1-0" if is_white_turn else "0-1"

    def _get_PGN_game(self, board: chess.Board, result: str) -> chess.pgn.Game:
        game = chess.pgn.Game(
            headers={
                "Event": self.event,
                "Site": self.site,
                "Date": self.date,
                "Round": self.round,
                "White": self.white_player,
                "Black": self.black_player,
                "Result": result,
                "WhiteElo": str(self.white_player_elo),
                "BlackElo": str(self.black_player_elo),
            }
        )

        node = game
        for move in board.move_stack:
            node = node.add_variation(move)

        return game

    def _convert_PGN_game_to_string(
        self,
        game: chess.pgn.Game,
        board: chess.Board,
        result: str,
    ) -> str:
        """
        Convert the PGN game to a string suitable for prompting an LLM.

        Args:
            game: The PGN game object
            board: The current chess board state
            result: The result of the game (e.g., "1-0", "0-1", "1/2-1/2")

        Returns:
            A string of the formatted PGN game
        """
        # Remove the trailing result characters
        game_str = str(game)[: -len(result)]

        # Format each full turn as its own line
        game_str = re.sub(r" (\d+)\. ", r"\n\1. ", game_str)

        if board.turn == chess.WHITE:  # Add current move number indicator
            n = board.fullmove_number
            if n == 1:
                game_str += "1."
            else:
                game_str += f"\n{n}."

        return game_str.strip()

    def build_prompt(self, board: chess.Board) -> str:
        """
        Build a prompt based on the current board position in PGN format.

        Args:
            board: The current chess board state

        Returns:
            A string containing a PGN representation of the game so far with
            appropriate formatting for prompting an LLM
        """
        result = self._get_result(board.turn == chess.WHITE)
        game = self._get_PGN_game(board, result)
        game_str = self._convert_PGN_game_to_string(game, board, result)
        return game_str if self.move_response_has_leading_space else game_str + " "

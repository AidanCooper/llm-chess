import logging
from datetime import datetime
from pathlib import Path

import chess

from llm_chess.core.game_manager import GameManager
from llm_chess.core.player import ChessPlayer
from llm_chess.players.engine.stockfish import StockfishPlayer
from llm_chess.utils.write import write_board_to_pgn_file

logger = logging.getLogger(__name__)


def calculate_expected_score(elo_a: float, elo_b: float) -> float:
    """
    Calculates the expected score of player A against player B based on their ELO ratings.

    Args:
        elo_a: ELO rating of player A.
        elo_b: ELO rating of player B.

    Returns:
        The expected score for player A (a float between 0 and 1).
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def update_elo(
    current_elo: float, opponent_elo: float, actual_score: float, k_factor: int = 32
) -> int:
    """
    Updates a player's ELO rating based on a game outcome.

    Args:
        current_elo: The player's current ELO rating.
        opponent_elo: The opponent's ELO rating.
        actual_score: The actual score obtained by the player (1 for win, 0.5 for draw, 0 for loss).
        k_factor: The K-factor, which determines the maximum possible ELO change per game.

    Returns:
        The player's new ELO rating.
    """
    expected_score = calculate_expected_score(current_elo, opponent_elo)
    new_elo = current_elo + k_factor * (actual_score - expected_score)
    return int(new_elo)


def calibrate_elo(
    player_to_calibrate: ChessPlayer,
    initial_llm_elo_estimate: int,
    num_games: int,
    start_k_factor: int = 32,
    end_k_factor: int = 16,
    stockfish_min_elo: int = 1320,
    stockfish_max_elo: int = 3190,
    write_dir: Path | None = None,
    n_randomised_starting_half_moves: int = 0,
) -> list[tuple[float, int]]:
    """
    Calibrates the ChessPlayer's ELO rating by playing against Stockfish.

    Args:
        player_to_calibrate: The ChessPlayer being calibrated.
        initial_llm_elo_estimate: The starting estimated ELO for the LLM.
        num_games: The total number of games to play for calibration.
        start_k_factor: The K-factor for the first game. The K-factor is then reduced
            linearly to the `end_k_factor` value for the final game.
        end_k_factor: The K-factor for the final game.
        stockfish_min_elo: The minimum ELO Stockfish can be configured to.
        stockfish_max_elo: The maximum ELO Stockfish can be configured to.
        write_dir: Optionally log games as PGN files to this directory.
        n_randomised_starting_half_moves: Number of random moves to make before before
            using the players' strategies.

    Returns:
        A list containing the (player score, player ELO) for each played game.
    """
    current_elo = initial_llm_elo_estimate
    stockfish_elo = initial_llm_elo_estimate  # Stockfish starts at the Player's estimated ELO

    logger.info("--- Starting Player ELO Calibration ---")
    logger.info(f"Initial Player ELO    : {current_elo}")
    logger.info(f"Initial Stockfish ELO : {stockfish_elo}")
    logger.info(f"Number of Games       : {num_games}")
    logger.info(f"K-Factor              : {start_k_factor} to {end_k_factor}")
    logger.info("-" * 35)

    game_scores_and_elos: list[tuple[float, int]] = []

    for i in range(1, num_games + 1):
        logger.info(f"\nGame {i}/{num_games}:")

        if stockfish_elo < stockfish_min_elo:
            logger.warning(f"  Minimum stockfish ELO reached ({stockfish_min_elo})")
            current_stockfish_elo = stockfish_min_elo
        elif stockfish_elo > stockfish_max_elo:
            logger.warning(f"  Maximum stockfish ELO reached ({stockfish_max_elo})")
            current_stockfish_elo = stockfish_max_elo
        else:
            logger.info(f"  Setting Stockfish ELO to: {stockfish_elo}")
            current_stockfish_elo = stockfish_elo
        stockfish_player = StockfishPlayer(
            f"Stockfish (ELO: {current_stockfish_elo})", elo=current_stockfish_elo
        )

        # Play the game
        player_plays_white = bool(i % 2)
        white, black = (
            (player_to_calibrate, stockfish_player)
            if player_plays_white
            else (stockfish_player, player_to_calibrate)
        )
        manager = GameManager()
        board = chess.Board()
        board, result = manager.play_game(
            white,
            black,
            board=board,
            displayer=None,
            sleep_time=0.2,
            n_randomised_starting_half_moves=n_randomised_starting_half_moves,
        )

        # Log the game
        if write_dir is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"{timestamp}_game_{i:05d}.pgn"
            write_board_to_pgn_file(
                board=board,
                write_dir=write_dir,
                file_name=file_name,
                white_name=white.name,
                black_name=black.name,
                result=result,
            )

        # Update Player's ELO based on the game outcome
        if result == "1/2-1/2":
            actual_score = 0.5
        elif white == player_to_calibrate:
            actual_score = 1.0 if result == "1-0" else 0.0
        else:
            actual_score = 1.0 if result == "0-1" else 0.0

        previous_elo = current_elo
        k_factor = int(start_k_factor - (start_k_factor - end_k_factor) * (i - 1) / (num_games - 1))
        current_elo = update_elo(current_elo, current_stockfish_elo, actual_score, k_factor)
        game_scores_and_elos.append((actual_score, current_elo))

        logger.info(
            f"  Player played {'white' if player_plays_white else 'black'}. Game result: {result}"
        )
        logger.info(
            f"  Player ELO updated from {previous_elo} to {current_elo} (K-Factor: {k_factor})"
        )

        # Adjust Stockfish's ELO for the next game to match the Player's current estimate
        stockfish_elo = current_elo

    logger.info("\n--- Calibration Complete ---")
    logger.info(f"Final Estimated Player ELO: {current_elo}")
    return game_scores_and_elos

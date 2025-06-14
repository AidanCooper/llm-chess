from unittest.mock import MagicMock, patch

import pytest

from llm_chess.utils import calibrate


def test_calculate_expected_score() -> None:
    assert calibrate.calculate_expected_score(1500, 1500) == pytest.approx(0.5)
    assert calibrate.calculate_expected_score(2000, 1500) > 0.9
    assert calibrate.calculate_expected_score(1000, 1500) < 0.1


def test_update_elo_win() -> None:
    new_elo = calibrate.update_elo(1500, 1500, 1.0, k_factor=32)
    assert new_elo > 1500


def test_update_elo_draw() -> None:
    new_elo = calibrate.update_elo(1500, 1500, 0.5, k_factor=32)
    assert new_elo == 1500


def test_update_elo_loss() -> None:
    new_elo = calibrate.update_elo(1500, 1500, 0.0, k_factor=32)
    assert new_elo < 1500


@patch("llm_chess.utils.calibrate.StockfishPlayer")
@patch("llm_chess.utils.calibrate.GameManager")
def test_calibrate_elo_basic(mock_gamemanager: MagicMock, mock_stockfishplayer: MagicMock) -> None:
    mock_player = MagicMock()
    mock_player.name = "TestPlayer"
    mock_stockfish = MagicMock()
    mock_stockfish.name = "Stockfish"
    mock_stockfishplayer.return_value = mock_stockfish

    mock_manager = MagicMock()
    results = [
        ("board", "1-0"),  # Player wins as white
        ("board", "0-1"),  # Player wins as black
        ("board", "1/2-1/2"),  # Player draws as white
        ("board", "1-0"),  # Player loses as black
    ]
    mock_manager.play_game.side_effect = results
    mock_gamemanager.return_value = mock_manager

    scores_and_elos = calibrate.calibrate_elo(
        player_to_calibrate=mock_player,
        initial_llm_elo_estimate=1500,
        num_games=len(results),
        start_k_factor=32,
        end_k_factor=16,
        write_dir=None,
    )
    assert len(scores_and_elos) == len(results)

    assert scores_and_elos[0][0] == 1.0
    assert scores_and_elos[0][1] > 1500

    assert scores_and_elos[1][0] == 1.0
    assert scores_and_elos[1][1] > scores_and_elos[0][1]

    assert scores_and_elos[2][0] == 0.5
    assert scores_and_elos[2][1] == scores_and_elos[1][1]

    assert scores_and_elos[3][0] == 0.0
    assert scores_and_elos[3][1] < scores_and_elos[2][1]

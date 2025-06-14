import chess
import pytest
from pytest import FixtureRequest

from llm_chess.prompts.pgn import PGNPromptConfig


@pytest.mark.parametrize(
    "board, move_response_has_leading_space, white_player, black_player, moves,"
    " expected_moves, result, expected_result",
    [
        (
            "starting_board",
            True,
            "Test Player A",
            "Test Player B",
            [],
            ["1."],
            "0-1",
            "0-1",
        ),
        (
            "starting_board",
            False,
            "Test Player A",
            "Test Player B",
            [],
            ["1. "],
            "0-1",
            "0-1",
        ),
        (
            "starting_board",
            True,
            "Test Player A",
            "Test Player B",
            ["e4"],
            ["1. e4"],
            None,
            "0-1",  # black's turn, therefore indicate a black win
        ),
        (
            "starting_board",
            False,
            "Test Player A",
            "Test Player B",
            ["e4"],
            ["1. e4 "],
            None,
            "0-1",  # black's turn, therefore indicate a black win
        ),
        (
            "starting_board",
            True,
            "Test Player A",
            "Test Player B",
            ["e4", "e5"],
            ["1. e4 e5", "2."],
            None,
            "1-0",  # white's turn, therefore indicate a white win
        ),
        (
            "starting_board",
            False,
            "Test Player A",
            "Test Player B",
            ["e4", "e5"],
            ["1. e4 e5", "2. "],
            None,
            "1-0",  # white's turn, therefore indicate a white win
        ),
        (
            "starting_board",
            True,
            "Test Player A",
            "Test Player B",
            ["e4", "e5", "Nf3"],
            ["1. e4 e5", "2. Nf3"],
            "1/2-1/2",
            "1/2-1/2",
        ),
        (
            "starting_board",
            False,
            "Test Player A",
            "Test Player B",
            ["e4", "e5", "Nf3"],
            ["1. e4 e5", "2. Nf3 "],
            "1/2-1/2",
            "1/2-1/2",
        ),
    ],
)
def test_pgn_prompt_config(
    board: str,
    move_response_has_leading_space: bool,
    white_player: str,
    black_player: str,
    result: str | None,
    moves: list[str],
    expected_result: str,
    expected_moves: list[str],
    request: FixtureRequest,
) -> None:
    board_fixture: chess.Board = request.getfixturevalue(board)
    for move in moves:
        board_fixture.push_san(move)

    config = PGNPromptConfig(
        move_response_has_leading_space=move_response_has_leading_space,
        white_player=white_player,
        black_player=black_player,
        result=result,
    )
    prompt = config.build_prompt(board_fixture)
    assert f'[White "{white_player}"]' in prompt
    assert f'[Black "{black_player}"]' in prompt
    assert f'[Result "{expected_result}"]' in prompt
    for expected_move in expected_moves:
        assert expected_move in prompt

from typing import Literal

import pytest

from llm_chess.core.enums import MoveNotation
from llm_chess.prompts.text_board import TextBoardPromptConfig, board_to_text


@pytest.mark.parametrize(
    "board, flip_board, piece_symbols, expected_output",
    [
        ("starting_board", False, True, "starting_board_str"),
        ("starting_board", False, False, "starting_board_str_without_symbols"),
        ("starting_board_black", True, True, "starting_board_str_black"),
        (
            "starting_board_black",
            True,
            False,
            "starting_board_str_black_without_symbols",
        ),
    ],
)
def test_board_to_text(
    board: str,
    flip_board: bool,
    piece_symbols: bool,
    expected_output: str,
    request: pytest.FixtureRequest,
) -> None:
    board_fixture = request.getfixturevalue(board)
    expected_output_fixture = request.getfixturevalue(expected_output)
    assert (
        board_to_text(board_fixture, flip_board=flip_board, piece_symbols=piece_symbols)
        == expected_output_fixture
    )


@pytest.mark.parametrize(
    "board, moves_formatted, colour, move_notation, piece_symbols, board_str",
    [
        (
            "starting_board",
            "legal_starting_moves_formatted_uci",
            "white",
            MoveNotation.UCI,
            True,
            "starting_board_str",
        ),
        (
            "starting_board",
            "legal_starting_moves_formatted_san",
            "white",
            MoveNotation.SAN,
            False,
            "starting_board_str_without_symbols",
        ),
        (
            "starting_board_black",
            "legal_starting_moves_formatted_black_uci",
            "black",
            MoveNotation.UCI,
            True,
            "starting_board_str_black",
        ),
        (
            "starting_board_black",
            "legal_starting_moves_formatted_black_san",
            "black",
            MoveNotation.SAN,
            False,
            "starting_board_str_black_without_symbols",
        ),
    ],
)
def test_text_board_prompt_config(
    board: str,
    moves_formatted: str,
    colour: Literal["white", "black"],
    move_notation: MoveNotation,
    piece_symbols: bool,
    board_str: str,
    request: pytest.FixtureRequest,
) -> None:
    board_fixture = request.getfixturevalue(board)
    moves_formatted_fixture = request.getfixturevalue(moves_formatted)
    board_str_fixture = request.getfixturevalue(board_str)
    config = TextBoardPromptConfig(move_notation=move_notation, piece_symbols=piece_symbols)
    prompt = config.build_prompt(board_fixture)
    assert f"It is your turn as {colour}." in prompt
    assert board_str_fixture in prompt
    assert moves_formatted_fixture in prompt

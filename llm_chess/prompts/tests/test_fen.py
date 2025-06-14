from typing import Literal

import chess
import pytest
from pytest import FixtureRequest

from llm_chess.core.enums import MoveNotation
from llm_chess.prompts.fen import FENPromptConfig


@pytest.mark.parametrize(
    "board, moves_formatted, colour, move_notation",
    [
        (
            "starting_board",
            "legal_starting_moves_formatted_uci",
            "white",
            MoveNotation.UCI,
        ),
        (
            "starting_board",
            "legal_starting_moves_formatted_san",
            "white",
            MoveNotation.SAN,
        ),
        (
            "starting_board_black",
            "legal_starting_moves_formatted_black_uci",
            "black",
            MoveNotation.UCI,
        ),
        (
            "starting_board_black",
            "legal_starting_moves_formatted_black_san",
            "black",
            MoveNotation.SAN,
        ),
    ],
)
def test_fen_prompt_config(
    board: str,
    moves_formatted: str,
    colour: Literal["white", "black"],
    move_notation: MoveNotation,
    request: FixtureRequest,
) -> None:
    board_fixture: chess.Board = request.getfixturevalue(board)
    moves_formatted_fixture: str = request.getfixturevalue(moves_formatted)
    config = FENPromptConfig(move_notation=move_notation)
    prompt = config.build_prompt(board_fixture)
    assert f"You are playing as {colour}." in prompt
    assert board_fixture.fen() in prompt
    assert moves_formatted_fixture in prompt

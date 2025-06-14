from typing import Any
from unittest.mock import Mock

import chess
import pytest

from llm_chess.players.llm.openai import OpenAIPlayer
from llm_chess.prompts.base import PromptConfig


@pytest.fixture
def mock_openai_response(content: str = '{"move": "e2e4"}') -> Mock:
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = content
    return response


@pytest.fixture
def mock_openai(monkeypatch: pytest.MonkeyPatch, mock_openai_response: Mock) -> Mock:
    mock_model = Mock()
    mock_model.chat.completions.create.return_value = mock_openai_response

    mock_openai_module = Mock()
    mock_openai_module.OpenAI.return_value = mock_model

    monkeypatch.setattr("llm_chess.players.llm.openai.openai", mock_openai_module)
    return mock_openai_module


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    test_key = "test_key"
    monkeypatch.setenv("OPENAI_API_KEY", test_key)
    return test_key


@pytest.fixture
def player(api_key: str, mock_openai: Mock, mock_prompt_config: PromptConfig) -> OpenAIPlayer:
    return OpenAIPlayer(name="TestOpenAI", prompt_config=mock_prompt_config)


@pytest.mark.parametrize(
    "name,model,supplied_key,expected",
    [
        (
            "TestPlayer",
            "model-1",
            None,
            {"name": "TestPlayer", "model": "model-1"},
        ),
        (
            "Player2",
            "model-2",
            "supplied-key",
            {"name": "Player2", "model": "model-2"},
        ),
    ],
)
def test_initialization(
    api_key: str,
    mock_openai: Mock,
    name: str,
    model: str,
    mock_prompt_config: PromptConfig,
    supplied_key: str | None,
    expected: dict[str, Any],
) -> None:
    player = OpenAIPlayer(
        name=name,
        model=model,
        prompt_config=mock_prompt_config,
        api_key=supplied_key or api_key,
    )

    assert player.name == expected["name"]
    assert player.model == expected["model"]
    mock_openai.OpenAI.assert_called_once_with(
        api_key=supplied_key or api_key, base_url="https://api.openai.com/v1"
    )


def test_model_response_success(player: OpenAIPlayer, starting_board: chess.Board) -> None:
    move = player._get_model_response(starting_board)
    assert move == "e2e4"


def test_model_response_handles_api_error(
    player: OpenAIPlayer, starting_board: chess.Board
) -> None:
    player.client.chat.completions.create.side_effect = Exception("API Error")
    with pytest.raises(RuntimeError, match="Error during API call"):
        player._get_model_response(starting_board)


def test_make_move_valid(player: OpenAIPlayer, starting_board: chess.Board) -> None:
    move = player.make_move(starting_board)
    assert isinstance(move, chess.Move)
    assert move in starting_board.legal_moves


def test_make_move_invalid(
    mock_openai_response: Mock, player: OpenAIPlayer, starting_board: chess.Board
) -> None:
    mock_openai_response.choices[0].message.content = '{"move": "invalid"}'

    with pytest.raises(ValueError):
        player.make_move(starting_board)

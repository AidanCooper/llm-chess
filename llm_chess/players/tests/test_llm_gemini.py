from unittest.mock import Mock

import chess
import pytest

from llm_chess.players.llm.gemini import GeminiPlayer


@pytest.fixture
def mock_gemini_response(content: str = '{"move": "e2e4"}') -> Mock:
    response = Mock()
    response.text = content
    return response


@pytest.fixture
def mock_genai(monkeypatch: pytest.MonkeyPatch, mock_gemini_response: Mock) -> Mock:
    mock_model = Mock()
    mock_model.generate_content.return_value = mock_gemini_response

    mock_genai_module = Mock()
    mock_genai_module.GenerativeModel.return_value = mock_model

    monkeypatch.setattr("llm_chess.players.llm.gemini.genai", mock_genai_module)
    return mock_genai_module


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    test_key = "test_key"
    monkeypatch.setenv("GEMINI_API_KEY", test_key)
    return test_key


@pytest.fixture
def player(api_key: str, mock_genai: Mock, mock_prompt_config: Mock) -> GeminiPlayer:
    return GeminiPlayer(name="TestGemini", prompt_config=mock_prompt_config)


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
    mock_genai: Mock,
    name: str,
    model: str,
    mock_prompt_config: Mock,
    supplied_key: str | None,
    expected: dict[str, str | int],
) -> None:
    from llm_chess.players.llm.gemini import GeminiPlayer

    player = GeminiPlayer(name=name, model=model, prompt_config=mock_prompt_config)

    assert player.name == expected["name"]
    assert player.api_key == supplied_key or api_key
    mock_genai.GenerativeModel.assert_called_once_with(model_name=expected["model"])


def test_model_response_success(player: GeminiPlayer, starting_board: chess.Board) -> None:
    move = player._get_model_response(starting_board)
    assert move == "e2e4"


def test_model_response_handles_api_error(
    player: GeminiPlayer, mock_genai: Mock, starting_board: chess.Board
) -> None:
    mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception("API Error")
    with pytest.raises(RuntimeError, match="Error during API call"):
        player._get_model_response(starting_board)


def test_make_move_valid(player: GeminiPlayer, starting_board: chess.Board) -> None:
    move = player.make_move(starting_board)
    assert isinstance(move, chess.Move)
    assert move in starting_board.legal_moves


def test_make_move_invalid(
    mock_gemini_response: Mock, player: GeminiPlayer, starting_board: chess.Board
) -> None:
    mock_gemini_response.text = "invalid"

    with pytest.raises(ValueError):
        player.make_move(starting_board)

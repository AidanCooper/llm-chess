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
def mock_genai_client(monkeypatch: pytest.MonkeyPatch, mock_gemini_response: Mock) -> Mock:
    mock_models = Mock()
    mock_models.generate_content.return_value = mock_gemini_response

    mock_client = Mock()
    mock_client.models = mock_models

    mock_genai_module = Mock()
    mock_genai_module.Client.return_value = mock_client

    monkeypatch.setattr("llm_chess.players.llm.gemini.genai", mock_genai_module)
    return mock_genai_module


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    test_key = "test_key"
    monkeypatch.setenv("GEMINI_API_KEY", test_key)
    return test_key


@pytest.fixture
def player(api_key: str, mock_genai_client: Mock, mock_prompt_config: Mock) -> GeminiPlayer:
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
    mock_genai_client: Mock,
    name: str,
    model: str,
    mock_prompt_config: Mock,
    supplied_key: str | None,
    expected: dict[str, str | int],
) -> None:
    from llm_chess.players.llm.gemini import GeminiPlayer

    if supplied_key:
        player = GeminiPlayer(
            name=name, model=model, prompt_config=mock_prompt_config, api_key=supplied_key
        )
    else:
        player = GeminiPlayer(name=name, model=model, prompt_config=mock_prompt_config)

    assert player.name == expected["name"]
    assert player.api_key == supplied_key or api_key
    mock_genai_client.Client.assert_called_once_with(api_key=supplied_key or api_key)


def test_model_response_success(player: GeminiPlayer, starting_board: chess.Board) -> None:
    move = player._get_model_response(starting_board)
    assert move == "e2e4"


def test_model_response_handles_api_error(
    player: GeminiPlayer, mock_genai_client: Mock, starting_board: chess.Board
) -> None:
    mock_genai_client.Client.return_value.models.generate_content.side_effect = Exception(
        "API Error"
    )
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


def test_structured_response_format(player: GeminiPlayer, starting_board: chess.Board) -> None:
    """Test that structured JSON response is parsed correctly."""
    move = player._get_model_response(starting_board)
    assert move == "e2e4"


def test_structured_response_invalid_json(
    mock_gemini_response: Mock, player: GeminiPlayer, starting_board: chess.Board
) -> None:
    """Test that invalid JSON in structured response raises ValueError."""
    mock_gemini_response.text = "invalid json"

    with pytest.raises(ValueError, match="Invalid response format from the model"):
        player._get_model_response(starting_board)


def test_structured_response_missing_move_key(
    mock_gemini_response: Mock, player: GeminiPlayer, starting_board: chess.Board
) -> None:
    """Test that missing 'move' key in JSON response raises ValueError."""
    mock_gemini_response.text = '{"other_key": "e2e4"}'

    with pytest.raises(ValueError, match="Invalid response format from the model"):
        player._get_model_response(starting_board)


def test_enum_response_format(
    mock_gemini_response: Mock,
    player: GeminiPlayer,
    mock_prompt_config: Mock,
    starting_board: chess.Board,
) -> None:
    """Test enum response format handling."""
    from llm_chess.core.enums import APIResponseFormat

    mock_prompt_config.api_response_format = APIResponseFormat.ENUM
    mock_gemini_response.text = "e2e4"

    move = player._get_model_response(starting_board)
    assert move == "e2e4"


def test_multi_turn_response_format(
    mock_genai_client: Mock, mock_prompt_config: Mock, starting_board: chess.Board
) -> None:
    """Test multi-turn response format handling."""
    from llm_chess.core.enums import APIResponseFormat

    mock_prompt_config.api_response_format = APIResponseFormat.MULTI_TURN
    mock_response = Mock()
    mock_response.text = "e2e4"
    mock_genai_client.Client.return_value.models.generate_content.return_value = mock_response

    player = GeminiPlayer(name="TestGemini", prompt_config=mock_prompt_config)
    move = player._get_model_response(starting_board)
    assert move == "e2e4"


def test_unsupported_response_format(mock_prompt_config: Mock, starting_board: chess.Board) -> None:
    """Test that unsupported response format raises ValueError."""
    mock_prompt_config.api_response_format = "UNSUPPORTED_FORMAT"

    player = GeminiPlayer(name="TestGemini", prompt_config=mock_prompt_config)

    with pytest.raises(ValueError, match="Unsupported API response format"):
        player._get_model_response(starting_board)

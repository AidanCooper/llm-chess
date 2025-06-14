import sys
from unittest.mock import Mock

import chess
import pytest

from llm_chess.core.enums import APIResponseFormat
from llm_chess.players.llm.sglang import SGLangPlayer


@pytest.fixture
def mock_sglang(monkeypatch: pytest.MonkeyPatch) -> Mock:
    import llm_chess.players.llm.sglang as sglang_module

    monkeypatch.setattr(sglang_module, "RuntimeEndpoint", lambda url: url)

    set_default_backend_mock = Mock()
    monkeypatch.setattr(sglang_module, "set_default_backend", set_default_backend_mock)

    def fake_function_decorator(func):
        class Wrapper:  # pylint: disable=too-few-public-methods
            def __init__(self, fn):
                self.fn = fn

            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

            @staticmethod
            def run(*_args, **_kwargs):
                return {"move": "e2e4"}

        return Wrapper(func)

    monkeypatch.setattr(sglang_module, "function", fake_function_decorator)

    for name in ("assistant", "system", "user", "gen"):
        monkeypatch.setattr(sglang_module, name, lambda *args, **kwargs: None)

    return set_default_backend_mock


@pytest.fixture
def player(mock_prompt_config, mock_sglang) -> SGLangPlayer:
    mock_prompt_config.api_response_format = APIResponseFormat.ENUM
    return SGLangPlayer(name="TestSGLang", prompt_config=mock_prompt_config)


@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux systems")
def test_initialization(mock_sglang, player) -> None:
    assert player.name == "TestSGLang"

    expected_url = "http://0.0.0.0:30000"
    mock_sglang.assert_called_once_with(expected_url)


@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux systems")
@pytest.mark.parametrize("response_format", [APIResponseFormat.ENUM, APIResponseFormat.MULTI_TURN])
def test_model_response_success(
    response_format: APIResponseFormat,
    mock_prompt_config,
    mock_sglang,
    player,
    starting_board: chess.Board,
) -> None:
    """_get_model_response should always forward the stubbed *e2e4* move."""
    move = player._get_model_response(starting_board)
    assert move == "e2e4"


@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux systems")
def test_model_response_handles_error(
    monkeypatch: pytest.MonkeyPatch,
    mock_prompt_config,
    mock_sglang,
    player,
    starting_board: chess.Board,
) -> None:
    """Player should surface *RuntimeError* when the handler fails."""
    # Force the handler to raise to simulate an internal failure.
    monkeypatch.setitem(
        player.response_handlers,
        APIResponseFormat.ENUM,
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("Error getting model response")),
    )

    with pytest.raises(RuntimeError, match="Error getting model response"):
        player._get_model_response(starting_board)


@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux systems")
def test_make_move_valid(player: SGLangPlayer, starting_board: chess.Board) -> None:
    """make_move converts the raw string into a valid `chess.Move`."""
    move = player.make_move(starting_board)

    assert isinstance(move, chess.Move)
    assert move in starting_board.legal_moves


@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux systems")
def test_make_move_invalid(
    player: SGLangPlayer,
    starting_board: chess.Board,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid engine reply should raise *ValueError*."""
    monkeypatch.setattr(player, "_get_model_response", lambda *_a, **_kw: "invalid")

    with pytest.raises(ValueError):
        player.make_move(starting_board)

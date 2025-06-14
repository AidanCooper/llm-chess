import chess
import pytest

from llm_chess.players.engine.stockfish import StockfishPlayer


@pytest.fixture
def player() -> StockfishPlayer:
    return StockfishPlayer(name="TestStockfish")


def test_initialization() -> None:
    player = StockfishPlayer(name="TestStockfish", elo=1320)
    assert player.name == "TestStockfish"
    assert player.elo == 1320
    assert player.config.elo == 1320


def test_make_move_valid(player: StockfishPlayer, starting_board: chess.Board) -> None:
    move = player.make_move(starting_board)
    assert isinstance(move, chess.Move)
    assert move in starting_board.legal_moves


def test_make_move_stalemate(player: StockfishPlayer, stalemate_board: chess.Board) -> None:
    move = player.make_move(stalemate_board)
    assert move is None


def test_engine_initialization(player: StockfishPlayer) -> None:
    assert player.engine is not None
    player.engine.quit()
    player.engine = None
    player.engine = player._initialize_engine(player.engine_path)
    assert player.engine is not None


def test_context_manager(player: StockfishPlayer) -> None:
    with player as p:
        assert p.engine is not None
    assert player.engine is None

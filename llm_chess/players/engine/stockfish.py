import os
from dataclasses import dataclass

import chess
import chess.engine

from llm_chess.core.player import ChessPlayer


@dataclass
class EngineConfig:
    elo: int = 1320  # 1320‑3190 (SF limit)
    movetime_ms: int = 1000  # per‑move wall‑time
    threads: int = 1
    hash_mb: int = 64


class StockfishPlayer(ChessPlayer):

    def __init__(
        self,
        name: str,
        engine_path: str | None = None,
        elo: int = 1320,
        movetime_ms: int = 1000,
        threads: int = 1,
        hash_mb: int = 64,
    ) -> None:
        assert elo >= 1320 and elo <= 3190, "Elo must be between 1320 and 3190."
        super().__init__(name)

        self.elo = elo
        self.engine_path = engine_path or os.getenv("STOCKFISH_ENGINE_PATH")
        self.config = EngineConfig(elo, movetime_ms, threads, hash_mb)
        self.engine = self._initialize_engine(self.engine_path)

    def _initialize_engine(self, engine_path: str | None) -> chess.engine.SimpleEngine:
        """Initialize and configure the chess engine."""
        if engine_path is None:
            raise ValueError(
                "STOCKFISH_ENGINE_PATH must be set in the environment or passed as argument."
            )
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        engine.configure(
            {
                "Threads": self.config.threads,
                "Hash": self.config.hash_mb,
                "UCI_LimitStrength": True,
                "UCI_Elo": self.config.elo,
            }
        )
        return engine

    def _get_move(
        self,
        board: chess.Board,
    ) -> chess.Move:
        if self.engine is None:
            self.engine = self._initialize_engine(self.engine_path)
        limit = chess.engine.Limit(time=self.config.movetime_ms / 1000)
        move = self.engine.play(board, limit).move
        if move is None:
            raise ValueError("Engine returned no move.")
        return move

    def __del__(self) -> None:
        """Cleanup: ensure engine process is terminated."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass

    def __enter__(self) -> "StockfishPlayer":
        """Context manager entry."""
        if not self.engine:
            self.engine = self._initialize_engine(self.engine_path)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit."""
        if self.engine:
            self.engine.quit()
            self.engine = None

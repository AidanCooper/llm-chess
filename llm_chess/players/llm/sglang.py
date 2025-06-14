import logging

import chess
from sglang import (
    RuntimeEndpoint,
    assistant,
    function,
    gen,
    set_default_backend,
    system,
    user,
)
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server

from llm_chess.core.enums import APIResponseFormat
from llm_chess.players.llm.base import LLMPlayer
from llm_chess.prompts.base import PromptConfig
from llm_chess.utils.format import format_legal_moves, format_moves_history

logger = logging.getLogger(__name__)


class SGLangPlayer(LLMPlayer):
    """
    A player that uses a local model served using SGLang.

    Supplying a model path will start a local server to serve the model. If no model
    path is supplied, the player will assume the server is already running.
    """

    def __init__(
        self,
        name: str,
        prompt_config: PromptConfig,
        model_path: str | None = None,
        host: str = "0.0.0.0",
        port: int | None = 30000,
    ):
        super().__init__(name, prompt_config)
        self.model_path = model_path
        self.host = host
        self.port = port
        self.server_process = None
        if self.model_path:
            try:
                self.server_process, self.port = self._initialize_server(
                    self.model_path, self.host, self.port
                )
            except Exception as e:
                logger.error(f"Failed to initialize server: {e}")
                raise
        set_default_backend(RuntimeEndpoint(f"http://{self.host}:{port}"))

        self.response_handlers = {
            APIResponseFormat.STRUCTURED: self._handle_enum_response,
            APIResponseFormat.JSON: self._handle_enum_response,
            APIResponseFormat.ENUM: self._handle_enum_response,
            APIResponseFormat.TEXT: self._handle_enum_response,
            APIResponseFormat.MULTI_TURN: self._handle_multi_turn_response,
        }

    def _initialize_server(self, model_path: str, host: str, port: int | None) -> tuple:
        try:
            server_process, port = launch_server_cmd(
                f"python -m sglang.launch_server --model-path {model_path}",
                host=self.host,
                port=self.port,
            )
            wait_for_server(f"http://{self.host}:{port}")
            logger.info(f"Server initialized at http://{self.host}:{port}")
            return server_process, port
        except Exception as e:
            logger.error(f"Error initializing server: {e}")
            raise

    def _get_model_response(self, board: chess.Board) -> str:
        prompt = self.prompt_config.build_prompt(board)
        try:
            handler = self.response_handlers[self.prompt_config.api_response_format]
        except KeyError as e:
            raise ValueError(
                f"Unsupported API response format: {self.prompt_config.api_response_format}"
            ) from e
        return handler(prompt, board)

    def _handle_multi_turn_response(self, prompt: str, board: chess.Board) -> str:
        notation = self.prompt_config.move_notation
        formatted_moves_history = format_moves_history(board, notation)
        formatted_legal_moves = format_legal_moves(board, notation)

        @function
        def select_move(s, prompt: str, model_is_white: bool, formatted_legal_moves: list[str]):
            s += system("You are a chess engine.")
            s += user(prompt)
            if not model_is_white:
                s += assistant("Understood. Let's play.")
            for idx, move in enumerate(formatted_moves_history):
                is_model_turn = (idx % 2 == 0) if model_is_white else (idx % 2 != 0)
                s += assistant(move) if is_model_turn else user(move)
            s += assistant(gen("move", choices=formatted_legal_moves))

        try:
            state = select_move.run(
                prompt=prompt,
                model_is_white=len(formatted_moves_history) % 2 == 0,
                formatted_legal_moves=formatted_legal_moves,
            )
            return state["move"]
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            raise RuntimeError("Error getting model response") from e

    def _handle_enum_response(self, prompt: str, board: chess.Board) -> str:
        notation = self.prompt_config.move_notation
        legal_moves = format_legal_moves(board, notation)

        @function
        def select_move(s, prompt: str, formatted_legal_moves: list[str]):
            s += system("You are a chess engine.")
            s += user(prompt)
            s += assistant(gen("move", choices=formatted_legal_moves))

        try:
            state = select_move.run(prompt=prompt, formatted_legal_moves=legal_moves)
            return state["move"]
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            raise RuntimeError("Error getting model response") from e

    def __enter__(self):
        """Context manager entry."""
        if self.model_path and not self.server_process:
            try:
                self.server_process, self.port = self._initialize_server(
                    self.model_path, self.host, self.port
                )
            except Exception as e:
                logger.error(f"Failed to initialise server in context manager: {e}")
                raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.server_process:
            try:
                terminate_process(self.server_process)
                logger.info("Server process terminated on exit.")
            except Exception as e:
                logger.warning(f"Failed to terminate process cleanly: {e}")
            finally:
                self.server_process = None

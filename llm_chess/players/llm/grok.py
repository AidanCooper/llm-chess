import os
from enum import Enum

import chess
from pydantic import BaseModel, Field
from xai_sdk import Client
from xai_sdk.chat import user

from llm_chess.core.enums import APIResponseFormat
from llm_chess.players.llm.base import LLMPlayer
from llm_chess.prompts.base import PromptConfig
from llm_chess.utils.format import format_legal_moves


class GrokPlayer(LLMPlayer):
    """
    Note: Grok can also be accessed via the OpenAI API as per the following code
    snippet. There seem to be some minor differences in the feature set and behaviour,
    however, hence the creation of this separate class that uses the xai_sdk package.

    ```python
    from llm_chess.players.llm.openai import OpenAIPlayer

    grok = OpenAIPlayer(
        name="Grok",
        model="grok-3-mini-fast",
        api_key=os.getenv("GROK_API_KEY"),
        base_url="https://api.x.ai/v1",
        prompt_config=pgn_prompt_config,
    )
    ```
    """

    def __init__(
        self,
        name: str,
        prompt_config: PromptConfig,
        model: str = "grok-3-mini-fast",
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: int = 3600,
    ):
        super().__init__(name, prompt_config)
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        if self.api_key is None:
            raise ValueError("GROK_API_KEY must be set in the environment or passed as argument.")
        self.client = Client(api_key=self.api_key, timeout=timeout)
        self.chat = self.client.chat.create(model=model, temperature=temperature)

        self.response_handlers = {
            # APIResponseFormat.STRUCTURED: self._handle_structured_response,
            # APIResponseFormat.JSON: self._handle_structured_response,
            APIResponseFormat.ENUM: self._handle_enum_response,
            APIResponseFormat.TEXT: self._handle_text_response,
            # APIResponseFormat.MULTI_TURN: self._handle_multi_turn_response,
        }

    def _get_model_response(self, board: chess.Board) -> str:
        prompt = self.prompt_config.build_prompt(board)
        try:
            handler = self.response_handlers[self.prompt_config.api_response_format]
        except KeyError as e:
            raise ValueError(
                f"Unsupported API response format: {self.prompt_config.api_response_format}"
            ) from e
        return handler(prompt, board)

    def _handle_text_response(self, prompt: str, board: chess.Board) -> str:
        self.chat.append(user(prompt))
        response = self.chat.sample()
        return str(response.content.strip())

    def _handle_enum_response(self, prompt: str, board: chess.Board) -> str:
        notation = self.prompt_config.move_notation
        formatted_legal_moves = format_legal_moves(board, notation)

        Move = Enum("Move", {move: move for move in formatted_legal_moves}, type=str)  # type: ignore

        class MoveResponse(BaseModel):  # type: ignore
            move: Move = Field(..., description=f"The move to play in {notation.value} format.")

        self.chat.append(user(prompt))
        _, move_response = self.chat.parse(MoveResponse)
        return str(move_response.move)

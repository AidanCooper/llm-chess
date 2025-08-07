import json
import os
from typing import Any

import chess
from google import genai
from google.genai import types

from llm_chess.core.enums import APIResponseFormat
from llm_chess.players.llm.base import LLMPlayer
from llm_chess.prompts.base import PromptConfig
from llm_chess.utils.format import format_legal_moves, format_moves_history


class GeminiPlayer(LLMPlayer):

    def __init__(
        self,
        name: str,
        prompt_config: PromptConfig,
        model: str = "gemini-2.0-flash-001",
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        super().__init__(name, prompt_config)
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key is None:
            raise ValueError("GEMINI_API_KEY must be set in the environment or passed as argument.")
        self.temperature = temperature
        self.client = genai.Client(api_key=self.api_key)

        self.response_handlers = {
            APIResponseFormat.STRUCTURED: self._handle_structured_response,
            APIResponseFormat.JSON: self._handle_structured_response,
            APIResponseFormat.ENUM: self._handle_enum_response,
            APIResponseFormat.TEXT: self._handle_enum_response,
            APIResponseFormat.MULTI_TURN: self._handle_multi_turn_response,
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

    def _handle_multi_turn_response(self, prompt: str, board: chess.Board) -> str:
        notation = self.prompt_config.move_notation
        formatted_moves_history = format_moves_history(board, notation)
        formatted_legal_moves = format_legal_moves(board, notation)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="text/x.enum",
            response_schema={"type": "STRING", "enum": formatted_legal_moves},
        )

        model_is_white = len(formatted_moves_history) % 2 == 0
        chat_history = [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]

        if not model_is_white:
            chat_history.append(
                {
                    "role": "model",
                    "parts": [{"text": "Understood. Let's play."}],
                }
            )

        for idx, move in enumerate(formatted_moves_history):
            is_model_turn = (idx % 2 == 0) if model_is_white else (idx % 2 != 0)
            author = "model" if is_model_turn else "user"
            chat_history.append({"role": author, "parts": [{"text": move}]})

        return self._call_model(chat_history, config)

    def _handle_enum_response(self, prompt: str, board: chess.Board) -> str:
        notation = self.prompt_config.move_notation
        legal_moves = format_legal_moves(
            board, notation, self.prompt_config.move_response_has_leading_space
        )

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="text/x.enum",
            response_schema={"type": "STRING", "enum": legal_moves},
        )
        return self._call_model(prompt, config)

    def _handle_structured_response(self, prompt: str, board: chess.Board) -> str:
        notation = self.prompt_config.move_notation
        legal_moves = format_legal_moves(
            board, notation, self.prompt_config.move_response_has_leading_space
        )

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {"move": {"type": "string", "enum": legal_moves}},
            },
        )
        response = self._call_model(prompt, config)
        try:
            return str(json.loads(response)["move"])
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError("Invalid response format from the model") from e

    def _call_model(
        self,
        contents: str | list[dict[str, Any]],
        generation_config: types.GenerateContentConfig,
    ) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config,
            )
            return str(response.text).strip()
        except Exception as e:
            raise RuntimeError(f"Error during API call: {e}") from e

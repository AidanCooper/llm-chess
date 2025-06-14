import json
import os
from typing import Any

import backoff
import chess
import openai

from llm_chess.core.enums import APIResponseFormat
from llm_chess.players.llm.base import LLMPlayer
from llm_chess.prompts.base import PromptConfig
from llm_chess.utils.format import format_legal_moves, format_moves_history


class OpenAIPlayer(LLMPlayer):

    def __init__(
        self,
        name: str,
        prompt_config: PromptConfig,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        base_url: str = "https://api.openai.com/v1",
    ):
        super().__init__(name, prompt_config)
        self.model = model
        self.temperature = temperature
        self.base_url = base_url

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("OPENAI_API_KEY must be set in the environment or passed as argument.")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.response_handlers = {
            APIResponseFormat.STRUCTURED: self._handle_structured_response,
            APIResponseFormat.JSON: self._handle_structured_response,
            APIResponseFormat.TEXT: self._handle_text_response,
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

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        model_is_white = len(formatted_moves_history) % 2 == 0
        if not model_is_white:
            messages.append(
                {
                    "role": "assistant",
                    "content": "Understood. Let's play.",
                }
            )
        for idx, move in enumerate(formatted_moves_history):
            is_model_turn = (idx % 2 == 0) if model_is_white else (idx % 2 != 0)
            author = "assistant" if is_model_turn else "user"
            messages.append({"role": author, "content": move})

        config = self._get_structured_response_config(board)
        response = self._call_model(messages, config)
        return str(response)

    def _handle_text_response(self, prompt: str, board: chess.Board) -> str:
        config = {"type": "text"}
        messages = [{"role": "user", "content": prompt}]
        response = self._call_model(messages, config)
        return str(response)

    def _handle_structured_response(self, prompt: str, board: chess.Board) -> str:
        config = self._get_structured_response_config(board)
        messages = [{"role": "user", "content": prompt}]
        response = self._call_model(messages, config)
        try:
            response_dict = json.loads(response)
            return str(response_dict["move"])
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError("Invalid response format from the model") from e

    def _get_structured_response_config(self, board: chess.Board) -> dict[str, Any]:
        notation = self.prompt_config.move_notation
        formatted_legal_moves = format_legal_moves(
            board, notation, self.prompt_config.move_response_has_leading_space
        )
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"{notation.value.lower()}_chess_move",
                "description": f"A valid chess move in {notation} format.",
                "strict": True,
                "schema": {
                    "type": "object",
                    "required": ["move"],
                    "properties": {
                        "move": {
                            "type": "string",
                            "enum": formatted_legal_moves,
                            "description": f"The chess move in {notation} format",
                        }
                    },
                    "additionalProperties": False,
                },
            },
        }

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    def _call_model(self, messages: list[dict[str, Any]], response_format: dict[str, Any]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format=response_format,
            )
            return str(response.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"Error during API call: {e}") from e

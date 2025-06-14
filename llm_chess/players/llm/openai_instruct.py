import logging
import os

import backoff
import chess
import openai

from llm_chess.players.llm.base import LLMPlayer
from llm_chess.prompts.base import PromptConfig
from llm_chess.prompts.pgn import PGNPromptConfig

PGN_PROMPT_CONFIG = PGNPromptConfig(move_response_has_leading_space=True)

logger = logging.getLogger(__name__)


class GPT3p5TurboInstructPlayer(LLMPlayer):

    def __init__(
        self,
        name: str = "GPT-3.5 Turbo Instruct",
        prompt_config: PromptConfig = PGN_PROMPT_CONFIG,
        model: str = "gpt-3.5-turbo-instruct",
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

    def _get_model_response(self, board: chess.Board) -> str:
        prompt = self.prompt_config.build_prompt(board)
        return str(self._call_model(prompt))

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    def _call_model(self, prompt: str, n_attempts: int = 3) -> str:
        for attempt in range(1, n_attempts + 1):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=7,
                )
            except Exception as e:
                raise RuntimeError(f"Error during API call: {e}") from e

            try:
                response_text = response.choices[0].text
                if response_text and response_text != "\n":
                    return str(response_text.strip().split()[0])
                logger.info(f"Invalid response on attempt {attempt}. Retrying...")
            except Exception as e:
                raise RuntimeError(f"Error during move extraction: {e}") from e

        raise RuntimeError("Model returned empty or invalid response after 3 attempts.")

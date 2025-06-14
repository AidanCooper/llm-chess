# llm-chess

My personal toolkit for exploring LLMs' chess playing abilities.

## Key Features

- **Modular Player Framework**: Implement various chess-playing agents, including LLM-based players and chess engines (Stockfish).
- **Multi-LLM Support**: Compatible with OpenAI (and OpenAI API compatible models, such as DeepSeek), Google Gemini, and SGLang models.
- **Chess Game Management**: Utilities for managing chess games, positions, and move validation. See the [game examples notebook](./usage_examples/game_examples.ipynb).
- **Prompt Engineering**: Customizable prompt templates for guiding LLMs in chess gameplay. See the [prompt examples notebook](./usage_examples/prompt_examples.ipynb).
- **Game Logging**: Comprehensive logging of games in PGN format.
- **Calibration Options**: Tools for calibrating LLM performance with different starting positions. See the [gpt-3.5-turbo-instruct calibration notebook](./usage_examples/GPT3p5TurboInstruct_ELO_calibration.ipynb).

## Installation

```bash
pip install -e '.[test]'
pre-commit install
```

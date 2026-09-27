# GPT-3.5 Turbo Instruct vs Frontier Chat Models (August 2026)

A rerun of [the August 2025 experiment](../gpt_3p5_turbo_instruct_vs_sota_reasoning_models),
one year on. Does a 2023 non-reasoning completion model still beat the frontier?

Yes. **GPT-3.5 Turbo Instruct won all four games. Every loss was an illegal move —
no game reached checkmate or resignation.**

## Setup

Opponents played through their **consumer chat interfaces** rather than the API, for two
reasons: no API cost for the frontier side, and each game gets a shareable conversation
as evidence. Moves were relayed between the browser and the `llm-chess` game state by
[`play_web_game.py`](../../play_web_game.py).

| Side | Model | Interface | Prompt |
|---|---|---|---|
| Instruct | `gpt-3.5-turbo-instruct`, `temperature=0`, `max_tokens=7` | `v1/completions` API | PGN continuation ([`PGNPromptConfig`](../../llm_chess/prompts/pgn.py)) |
| Opponents | GPT-5.6 Sol, Claude Opus 5, both at **medium** thinking effort | chatgpt.com / claude.ai | multi-turn (adapted setup in [`play_web_game.py`](../../play_web_game.py)) |

The opponent conversation opens with one setup message, after which every message in
both directions is intended to be a bare SAN move. An illegal or unparseable move is an
instant loss, as in 2025. The relay also treats an Instruct API/runtime error as a loss;
none of these four games ended that way.

Instruct receives the complete move history through `PGNPromptConfig` on every turn,
not the archive PGN verbatim. Its synthetic headers name Kasparov and Carlsen, both
rated 2800, at a World Championship in Moscow. The result header favours the side to
move. The prompt omits the trailing result and uses the formatter's precise whitespace.
The chat setup and PGN formatter are linked above so the input protocols can be inspected.

## Results

| # | White | Black | Result | Losing move |
|---|---|---|---|---|
| 1 | GPT-3.5 | GPT-5.6 Sol | **1-0** | 21...`Bxd6` — dark-squared bishop captured by 19.Rxd4; c8 bishop cannot reach d6 |
| 2 | GPT-3.5 | Claude Opus 5 | **1-0** | 20...`Kxc6` — c6 defended by the d5 pawn |
| 3 | GPT-5.6 Sol | GPT-3.5 | **0-1** | 21.`Bxd5` — only bishop on e3, no route to d5 |
| 4 | Claude Opus 5 | GPT-3.5 | **0-1** | 22.`Rxe3` — recapture with an already-traded rook |

Games: [1](game1_gpt5p6_black/game.pgn) · [2](game2_opus5_black/game.pgn) ·
[3](game3_gpt5p6_white/game.pgn) · [4](game4_opus5_white/game.pgn)

Conversations: [1](https://chatgpt.com/share/6a81ef28-4d78-83eb-920a-72094ad200f1) ·
[2](https://claude.ai/share/950ed941-2f0a-4c99-b74a-2da97ebbf73d) ·
[3](https://chatgpt.com/share/6a81ef3d-9c50-83eb-a06b-163e15f1029b) ·
[4](https://claude.ai/share/ff3d2104-02f6-476a-a62a-f5e70ca0c69e)

## Observations

- **Three attempted captures require a bishop or rook that cannot make the move.**
  They are consistent with overlooking an earlier capture, but the moves alone do not
  establish the models' internal representations. The fourth moves a king to a
  pawn-defended square.
- **The illegal attempts occur at moves 21, 20, 21 and 22.** Legal moves before that
  point do not by themselves demonstrate accurate internal board tracking.
- **Games 3 and 4 share the first 14 moves**, diverging at White's 15th move.
  Instruct used `temperature=0`; this does not guarantee deterministic API responses.
- `gpt-3.5-turbo-instruct` carries a **shutdown date of 2026-09-28**, so this is likely
  the last window in which this comparison can be run.

## Caveats

- **The chat surface is not the API.** Opponents ran with a system prompt, account
  personalisation, and tool access that the 2025 API runs did not have. In *both* of its
  games, GPT-5.6 Sol repeatedly tried to offload the position to an engine — importing
  `python-chess`, probing for a Stockfish binary (neither available in its sandbox) — and
  searched the web for opening lines. Claude Opus 5's games surface no reasoning or tool
  traces, so its tool use is unknown either way. The effect of these interface
  differences is not established; the two experiments are not like-for-like.
- **One game per pairing per colour.** Four games is an anecdote, not an Elo estimate.
- Thinking effort was left at the default **medium**. Higher effort was not tested.

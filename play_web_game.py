"""Drive a game between GPT-3.5 Turbo Instruct (API) and a chat-UI opponent.

The instruct player moves via the OpenAI completions API on the PGN prompt, exactly as in
`usage_examples/GPT3p5TurboInstruct_vs_SotA_Reasoning_Models.ipynb`. The opponent plays
in a ChatGPT/Claude web conversation, in the multi-turn style of
`llm_chess/prompts/multi_turn.py`: one setup message, then every message on both sides is
a single SAN move. `prompt` prints the next message to paste; `opp` records the reply.

Game state lives in a PGN file, so each command is a standalone process.

Usage:
    python play_web_game.py new  <game_id> --opponent "GPT-5.6 Sol" --instruct white
    python play_web_game.py prompt <game_id>        # text to paste into the chat UI
    python play_web_game.py opp    <game_id> Nf3    # record the chat UI's reply (SAN)
    python play_web_game.py instruct <game_id>      # query GPT-3.5 and play its move
    python play_web_game.py show   <game_id>
"""

import argparse
import json
from datetime import date
from pathlib import Path

import chess
import chess.pgn
from dotenv import load_dotenv

from llm_chess.players.llm.openai_instruct import GPT3p5TurboInstructPlayer

LOG_DIR = Path(__file__).parent / "game_logs" / "gpt_3p5_turbo_instruct_vs_chat_uis"

INSTRUCT_NAME = "GPT-3.5 Turbo Instruct"

# Opening message for the chat-UI opponent, adapted from MultiTurnPromptConfig for a
# chat interface: it establishes the game, then every subsequent message on both sides
# is a single SAN move and nothing else.
OPPONENT_SETUP_TEMPLATE = (
    "You are a chess engine, playing as {colour}. Using SAN notation, respond directly "
    "with the best move. We will play a full game in this conversation: every message I "
    "send will be my move in SAN notation, and every message you send must contain only "
    "your move in SAN notation and nothing else. {opening}"
)


def opponent_setup_message(colour: str) -> str:
    opening = (
        "Make your first move."
        if colour == "white"
        else "I will send my first move in my next message."
    )
    return OPPONENT_SETUP_TEMPLATE.format(colour=colour, opening=opening)


def last_san(board: chess.Board) -> str:
    """SAN of the most recently played move."""
    probe = board.copy()
    move = probe.pop()
    return probe.san(move)


def game_dir(game_id: str) -> Path:
    return LOG_DIR / game_id


def load_game(game_id: str) -> tuple[chess.Board, dict]:
    d = game_dir(game_id)
    meta = json.loads((d / "meta.json").read_text())
    board = chess.Board()
    with open(d / "game.pgn") as f:
        game = chess.pgn.read_game(f)
    if game is not None:
        for move in game.mainline_moves():
            board.push(move)
    return board, meta


def save_game(game_id: str, board: chess.Board, meta: dict) -> None:
    d = game_dir(game_id)
    game = chess.pgn.Game()
    game.headers.update(
        {
            "Event": "GPT-3.5 Turbo Instruct vs frontier chat models",
            "Site": "Local (opponent played via web UI)",
            "Date": meta["date"],
            "White": meta["white"],
            "Black": meta["black"],
            "Result": meta.get("result") or board.result(),
        }
    )
    for key, header in (
        ("opponent_url", "OpponentConversation"),
        ("opponent_effort", "OpponentEffort"),
        ("termination", "Termination"),
    ):
        if meta.get(key):
            game.headers[header] = meta[key]

    node: chess.pgn.GameNode = game
    for move in board.move_stack:
        node = node.add_variation(move)

    (d / "game.pgn").write_text(str(game) + "\n")
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def status_line(board: chess.Board, meta: dict) -> str:
    if meta.get("result"):
        return f"GAME OVER: {meta['result']} ({meta.get('termination', 'n/a')})"
    if board.is_game_over():
        return f"GAME OVER: {board.result()} ({board.outcome().termination.name})"  # type: ignore[union-attr]
    turn = "white" if board.turn == chess.WHITE else "black"
    who = meta["white"] if board.turn == chess.WHITE else meta["black"]
    return f"Move {board.fullmove_number}, {turn} to play: {who}"


def check_over(board: chess.Board, meta: dict) -> None:
    if board.is_game_over() and not meta.get("result"):
        outcome = board.outcome()
        meta["result"] = board.result()
        meta["termination"] = outcome.termination.name if outcome else "UNKNOWN"


def cmd_new(args: argparse.Namespace) -> None:
    d = game_dir(args.game_id)
    if d.exists():
        raise SystemExit(f"{d} already exists")
    d.mkdir(parents=True)
    instruct_is_white = args.instruct == "white"
    meta = {
        "game_id": args.game_id,
        "date": date.today().isoformat(),
        "white": INSTRUCT_NAME if instruct_is_white else args.opponent,
        "black": args.opponent if instruct_is_white else INSTRUCT_NAME,
        "instruct_colour": args.instruct,
        "opponent_colour": "black" if instruct_is_white else "white",
        "setup_sent": False,
        "opponent": args.opponent,
        "opponent_effort": args.effort,
        "opponent_url": args.url,
        "result": None,
        "termination": None,
    }
    save_game(args.game_id, chess.Board(), meta)
    print(f"Created {d}")
    print(status_line(chess.Board(), meta))


def cmd_prompt(args: argparse.Namespace) -> None:
    """Print the next message to paste into the chat UI."""
    board, meta = load_game(args.game_id)
    if meta.get("result"):
        raise SystemExit(status_line(board, meta))
    expected = chess.WHITE if meta["instruct_colour"] == "black" else chess.BLACK
    if board.turn != expected:
        raise SystemExit(f"Not the opponent's turn. {status_line(board, meta)}")

    if not meta.get("setup_sent"):
        message = opponent_setup_message(meta["opponent_colour"])
        if not args.repeat:
            meta["setup_sent"] = True
            save_game(args.game_id, board, meta)
    else:
        message = last_san(board)  # GPT-3.5's reply

    print(message)


def cmd_opp(args: argparse.Namespace) -> None:
    board, meta = load_game(args.game_id)
    if meta.get("result"):
        raise SystemExit(status_line(board, meta))
    expected = chess.WHITE if meta["instruct_colour"] == "black" else chess.BLACK
    if board.turn != expected:
        raise SystemExit(f"Not the opponent's turn. {status_line(board, meta)}")

    try:
        move = board.parse_san(args.san)
    except ValueError:
        # An illegal or malformed move is an instant loss, as in the 2025 experiment.
        meta["result"] = "0-1" if board.turn == chess.WHITE else "1-0"
        meta["termination"] = f"ILLEGAL MOVE by {meta['opponent']}: {args.san!r}"
        save_game(args.game_id, board, meta)
        print(f"ILLEGAL/UNPARSEABLE MOVE {args.san!r} — {meta['opponent']} loses.")
        print(status_line(board, meta))
        return

    board.push(move)
    check_over(board, meta)
    save_game(args.game_id, board, meta)
    print(f"{meta['opponent']} plays {args.san}")
    print(status_line(board, meta))


def cmd_instruct(args: argparse.Namespace) -> None:
    load_dotenv()
    board, meta = load_game(args.game_id)
    if meta.get("result"):
        raise SystemExit(status_line(board, meta))
    expected = chess.WHITE if meta["instruct_colour"] == "white" else chess.BLACK
    if board.turn != expected:
        raise SystemExit(f"Not GPT-3.5's turn. {status_line(board, meta)}")

    player = GPT3p5TurboInstructPlayer()
    try:
        move = player.make_move(board)
    except Exception as e:
        meta["result"] = "0-1" if board.turn == chess.WHITE else "1-0"
        meta["termination"] = f"ILLEGAL MOVE or error by {INSTRUCT_NAME}: {e}"
        save_game(args.game_id, board, meta)
        print(f"{INSTRUCT_NAME} failed: {e}")
        print(status_line(board, meta))
        return

    assert move is not None
    san = board.san(move)
    board.push(move)
    check_over(board, meta)
    save_game(args.game_id, board, meta)
    print(f"{INSTRUCT_NAME} plays {san}")
    print(status_line(board, meta))


def cmd_url(args: argparse.Namespace) -> None:
    """Record the shared conversation URL for the opponent's side of the game."""
    board, meta = load_game(args.game_id)
    meta["opponent_url"] = args.url
    save_game(args.game_id, board, meta)
    print(f"Recorded conversation URL: {args.url}")


def cmd_show(args: argparse.Namespace) -> None:
    board, meta = load_game(args.game_id)
    print(board.unicode(invert_color=True, empty_square="."))
    print()
    print((game_dir(args.game_id) / "game.pgn").read_text().strip())
    print()
    print(status_line(board, meta))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new")
    p_new.add_argument("game_id")
    p_new.add_argument("--opponent", required=True, help='e.g. "GPT-5.6 Sol"')
    p_new.add_argument("--instruct", choices=["white", "black"], required=True)
    p_new.add_argument("--effort", default=None, help="opponent thinking level")
    p_new.add_argument("--url", default=None, help="shared conversation URL")
    p_new.set_defaults(func=cmd_new)

    p_prompt = sub.add_parser("prompt")
    p_prompt.add_argument("game_id")
    p_prompt.add_argument(
        "--repeat",
        action="store_true",
        help="reprint the setup message without marking it as sent",
    )
    p_prompt.set_defaults(func=cmd_prompt)

    for name, func in (("instruct", cmd_instruct), ("show", cmd_show)):
        p = sub.add_parser(name)
        p.add_argument("game_id")
        p.set_defaults(func=func)

    p_url = sub.add_parser("url")
    p_url.add_argument("game_id")
    p_url.add_argument("url")
    p_url.set_defaults(func=cmd_url)

    p_opp = sub.add_parser("opp")
    p_opp.add_argument("game_id")
    p_opp.add_argument("san")
    p_opp.set_defaults(func=cmd_opp)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

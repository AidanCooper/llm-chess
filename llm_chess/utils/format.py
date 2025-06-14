import chess

from llm_chess.core.enums import MoveNotation


def convert_str_to_move(
    board: chess.Board, move_str: str, move_notation: MoveNotation
) -> chess.Move:
    """
    Parse a chess move from a string in the specified notation.

    Args:
        move_str (str): The chess move as a string.
        move_notation (MoveNotation): The notation to use for parsing.

    Returns:
        chess.Move: The parsed chess move.

    Raises:
        ValueError: If the move string is invalid or the notation is not supported.
    """
    try:
        if move_notation == MoveNotation.UCI:
            return board.parse_uci(move_str)
        if move_notation == MoveNotation.SAN:
            return board.parse_san(move_str)
        else:
            raise ValueError(f"Unsupported notation: {move_notation}")
    except chess.InvalidMoveError as e:
        raise chess.InvalidMoveError(f"Invalid move string: {move_str}. Error: {e}") from e


def convert_move_to_str(
    board: chess.Board,
    move: chess.Move,
    move_notation: MoveNotation = MoveNotation.UCI,
) -> str:
    """
    Convert a UCI chess move string to the specified notation.

    If the notation is UCI, the move is returned as is.

    Args:
        uci_str (str): The chess move in UCI notation
        move_notation (MoveNotation): The notation to use for formatting.

    Returns:
        str: The move in the specified notation.
    """
    try:
        if move_notation == MoveNotation.UCI:
            return str(board.uci(move))
        if move_notation == MoveNotation.SAN:
            return str(board.san(move))
        else:
            raise ValueError(f"Unsupported notation: {move_notation}")
    except chess.InvalidMoveError as e:
        raise chess.InvalidMoveError(f"Invalid move: {move}. Error: {e}") from e


def format_legal_moves(
    board: chess.Board, move_notation: MoveNotation, move_response_has_leading_space: bool = False
) -> list[str]:
    """
    Format the legal moves of a chess board in the specified notation. Typically used to
    yield the list of permitted LLM responses.

    Args:
        board (chess.Board): The chess board.
        move_notation (MoveNotation): The notation to use for formatting.
        move_response_has_leading_space (bool): Whether to add a leading space to the
            formatted moves.

    Returns:
        list[str]: A list of legal moves in the specified notation.
    """
    out = [convert_move_to_str(board, move, move_notation) for move in board.legal_moves]
    if move_response_has_leading_space:
        out = [f" {move}" for move in out]
    return out


def format_moves_history(board: chess.Board, move_notation: MoveNotation) -> list[str]:
    """
    Format the moves history of a chess board in the specified notation.

    Args:
        board (chess.Board): The chess board.
        move_notation (MoveNotation): The notation to use for formatting.

    Returns:
        list[str]: A list of moves in the specified notation in the order they were
            played for the supplied board.
    """
    board_ = chess.Board()
    formatted_moves = []
    for move in board.move_stack:
        formatted_moves.append(convert_move_to_str(board_, move, move_notation))
        board_.push(move)
    return formatted_moves

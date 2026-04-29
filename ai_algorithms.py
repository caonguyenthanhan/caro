from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from game_logic import CaroBoard, Move


PATTERN_SCORES = {
    "win": 100_000,
    "open4": 10_000,
    "blocked4": 1_000,
    "open3": 1_000,
    "blocked3": 100,
    "open2": 100,
}


def _count_overlapping(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    count = 0
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return count
        count += 1
        start = idx + 1


def _iter_lines(board: np.ndarray) -> list[str]:
    size = board.shape[0]
    lines: list[str] = []

    for r in range(size):
        lines.append("".join(str(int(x)) for x in board[r, :]))
    for c in range(size):
        lines.append("".join(str(int(x)) for x in board[:, c]))

    for offset in range(-size + 1, size):
        diag = np.diagonal(board, offset=offset)
        if diag.size >= 5:
            lines.append("".join(str(int(x)) for x in diag))
        anti = np.diagonal(np.fliplr(board), offset=offset)
        if anti.size >= 5:
            lines.append("".join(str(int(x)) for x in anti))

    return lines


def _score_player_lines(lines: list[str], player: int) -> int:
    p = str(player)
    o = str(3 - player)

    win = p * 5
    open4 = "0" + p * 4 + "0"
    open3 = "0" + p * 3 + "0"
    open2 = "0" + p * 2 + "0"

    score = 0
    for s in lines:
        if win in s:
            score += PATTERN_SCORES["win"] * _count_overlapping(s, win)

        score += PATTERN_SCORES["open4"] * _count_overlapping(s, open4)
        score += PATTERN_SCORES["open3"] * _count_overlapping(s, open3)
        score += PATTERN_SCORES["open2"] * _count_overlapping(s, open2)

        score += PATTERN_SCORES["blocked4"] * (
            _count_overlapping(s, o + p * 4 + "0")
            + _count_overlapping(s, "0" + p * 4 + o)
            + (1 if s.startswith(p * 4 + "0") else 0)
            + (1 if s.endswith("0" + p * 4) else 0)
        )

        score += PATTERN_SCORES["blocked3"] * (
            _count_overlapping(s, o + p * 3 + "0")
            + _count_overlapping(s, "0" + p * 3 + o)
            + (1 if s.startswith(p * 3 + "0") else 0)
            + (1 if s.endswith("0" + p * 3) else 0)
        )

    return score


def evaluate_board(board: CaroBoard, ai_player: int) -> int:
    lines = _iter_lines(board.board)
    opponent = 3 - ai_player

    ai_score = _score_player_lines(lines, ai_player)
    opp_score = _score_player_lines(lines, opponent)

    opp_win = (str(opponent) * 5) in "|".join(lines)
    opp_open4 = ("0" + str(opponent) * 4 + "0") in "|".join(lines)
    if opp_win:
        opp_score += 200_000
    elif opp_open4:
        opp_score += 50_000

    ai_win = (str(ai_player) * 5) in "|".join(lines)
    if ai_win:
        ai_score += 200_000

    return ai_score - opp_score


@dataclass(frozen=True)
class AiStats:
    nodes_evaluated: int
    time_taken_ms: float
    heuristic_score: int


class RandomAI:
    def choose_move(self, board: CaroBoard, ai_player: int = 2) -> tuple[Move | None, AiStats]:
        start = time.perf_counter()
        moves = board.get_possible_moves()
        if not moves:
            return None, AiStats(nodes_evaluated=0, time_taken_ms=0.0, heuristic_score=evaluate_board(board, ai_player))

        idx = int(np.random.randint(0, len(moves)))
        move = moves[idx]
        score = evaluate_board(board, ai_player)
        end = time.perf_counter()
        return move, AiStats(nodes_evaluated=1, time_taken_ms=(end - start) * 1000.0, heuristic_score=score)


class MinimaxAI:
    def __init__(self, depth: int = 2) -> None:
        self.depth = max(1, int(depth))

    def choose_move(self, board: CaroBoard, ai_player: int = 2) -> tuple[Move | None, AiStats]:
        start = time.perf_counter()
        nodes = 0

        def minimax(depth: int, current_player: int, last_move: Move | None) -> int:
            nonlocal nodes

            if last_move is not None:
                win = board.check_winner(last_move[0], last_move[1])
                if win is not None:
                    return 1_000_000 if win.player == ai_player else -1_000_000
            if depth == 0 or board.is_full():
                nodes += 1
                return evaluate_board(board, ai_player)

            maximizing = current_player == ai_player
            best = -10**18 if maximizing else 10**18
            for r, c in board.get_possible_moves():
                board.make_move(r, c, current_player)
                val = minimax(depth - 1, 3 - current_player, (r, c))
                board.undo_move(r, c)
                if maximizing:
                    if val > best:
                        best = val
                else:
                    if val < best:
                        best = val
            return int(best)

        best_move: Move | None = None
        best_val = -10**18
        for r, c in board.get_possible_moves():
            board.make_move(r, c, ai_player)
            val = minimax(self.depth - 1, 3 - ai_player, (r, c))
            board.undo_move(r, c)
            if val > best_val:
                best_val = val
                best_move = (r, c)

        score = evaluate_board(board, ai_player)
        end = time.perf_counter()
        return best_move, AiStats(nodes_evaluated=nodes, time_taken_ms=(end - start) * 1000.0, heuristic_score=score)


class AlphaBetaAI:
    def __init__(self, depth: int = 3) -> None:
        self.depth = max(1, int(depth))

    def choose_move(self, board: CaroBoard, ai_player: int = 2) -> tuple[Move | None, AiStats]:
        start = time.perf_counter()
        nodes = 0

        def ordered_moves(current_player: int) -> list[Move]:
            scored: list[tuple[int, Move]] = []
            for r, c in board.get_possible_moves():
                board.make_move(r, c, current_player)
                s = evaluate_board(board, ai_player)
                board.undo_move(r, c)
                scored.append((s, (r, c)))
            reverse = current_player == ai_player
            scored.sort(key=lambda x: x[0], reverse=reverse)
            return [m for _s, m in scored]

        def alphabeta(depth: int, current_player: int, alpha: int, beta: int, last_move: Move | None) -> int:
            nonlocal nodes

            if last_move is not None:
                win = board.check_winner(last_move[0], last_move[1])
                if win is not None:
                    return 1_000_000 if win.player == ai_player else -1_000_000
            if depth == 0 or board.is_full():
                nodes += 1
                return evaluate_board(board, ai_player)

            maximizing = current_player == ai_player
            if maximizing:
                value = -10**18
                for r, c in ordered_moves(current_player):
                    board.make_move(r, c, current_player)
                    value = max(value, alphabeta(depth - 1, 3 - current_player, alpha, beta, (r, c)))
                    board.undo_move(r, c)
                    alpha = max(alpha, int(value))
                    if alpha >= beta:
                        break
                return int(value)

            value = 10**18
            for r, c in ordered_moves(current_player):
                board.make_move(r, c, current_player)
                value = min(value, alphabeta(depth - 1, 3 - current_player, alpha, beta, (r, c)))
                board.undo_move(r, c)
                beta = min(beta, int(value))
                if alpha >= beta:
                    break
            return int(value)

        best_move: Move | None = None
        best_val = -10**18
        alpha = -10**18
        beta = 10**18

        for r, c in ordered_moves(ai_player):
            board.make_move(r, c, ai_player)
            val = alphabeta(self.depth - 1, 3 - ai_player, alpha, beta, (r, c))
            board.undo_move(r, c)
            if val > best_val:
                best_val = val
                best_move = (r, c)
            alpha = max(alpha, int(best_val))

        score = evaluate_board(board, ai_player)
        end = time.perf_counter()
        return best_move, AiStats(nodes_evaluated=nodes, time_taken_ms=(end - start) * 1000.0, heuristic_score=score)


def iter_search_demo(
    board: CaroBoard,
    algorithm: str,
    depth: int,
    ai_player: int,
    max_children: int = 12,
) -> tuple[Move | None, list[dict]]:
    max_children = max(1, int(max_children))
    depth = max(1, int(depth))

    events: list[dict] = []
    node_id_seq = 0

    def emit(ev: dict) -> None:
        events.append(ev)

    def limited_moves(current_player: int) -> list[Move]:
        moves = board.get_possible_moves()
        if not moves:
            return []
        scored: list[tuple[int, Move]] = []
        for r, c in moves:
            board.make_move(r, c, current_player)
            s = evaluate_board(board, ai_player)
            board.undo_move(r, c)
            scored.append((s, (r, c)))
        reverse = current_player == ai_player
        scored.sort(key=lambda x: x[0], reverse=reverse)
        return [m for _s, m in scored[:max_children]]

    def terminal_value(last_move: Move | None) -> int | None:
        if last_move is not None:
            win = board.check_winner(last_move[0], last_move[1])
            if win is not None:
                return 1_000_000 if win.player == ai_player else -1_000_000
        if board.is_full():
            return 0
        return None

    if algorithm.lower().startswith("mini"):
        def minimax_dfs() -> tuple[Move | None, int]:
            nonlocal node_id_seq
            INF = 10**18

            @dataclass
            class Frame:
                node_id: int
                parent_id: int | None
                depth_left: int
                current_player: int
                alpha: int
                beta: int
                last_move: Move | None
                moves: list[Move]
                i: int
                best_value: int
                best_move: Move | None
                incoming_move: Move | None

            def new_id() -> int:
                nonlocal node_id_seq
                node_id_seq += 1
                return node_id_seq

            root_id = new_id()
            root = Frame(
                node_id=root_id,
                parent_id=None,
                depth_left=depth,
                current_player=ai_player,
                alpha=-INF,
                beta=INF,
                last_move=board.last_move,
                moves=limited_moves(ai_player),
                i=0,
                best_value=-INF,
                best_move=None,
                incoming_move=None,
            )
            emit(
                {
                    "type": "enter",
                    "node_id": root.node_id,
                    "parent_id": None,
                    "depth": depth - root.depth_left,
                    "player": root.current_player,
                    "move": None,
                    "alpha": None,
                    "beta": None,
                    "value": None,
                    "note": "root",
                }
            )

            stack: list[Frame] = [root]
            while stack:
                fr = stack[-1]

                term = terminal_value(fr.last_move)
                if term is not None:
                    emit(
                        {
                            "type": "leaf",
                            "node_id": fr.node_id,
                            "parent_id": fr.parent_id,
                            "depth": depth - fr.depth_left,
                            "player": fr.current_player,
                            "move": fr.incoming_move,
                            "alpha": None,
                            "beta": None,
                            "value": int(term),
                            "note": "terminal",
                        }
                    )
                    stack.pop()
                    if fr.incoming_move is not None:
                        board.undo_move(fr.incoming_move[0], fr.incoming_move[1])
                    if stack:
                        parent = stack[-1]
                        maximizing = parent.current_player == ai_player
                        if maximizing:
                            if term > parent.best_value:
                                parent.best_value = int(term)
                                parent.best_move = fr.incoming_move
                        else:
                            if term < parent.best_value:
                                parent.best_value = int(term)
                                parent.best_move = fr.incoming_move
                    continue

                if fr.depth_left == 0:
                    val = evaluate_board(board, ai_player)
                    emit(
                        {
                            "type": "leaf",
                            "node_id": fr.node_id,
                            "parent_id": fr.parent_id,
                            "depth": depth - fr.depth_left,
                            "player": fr.current_player,
                            "move": fr.incoming_move,
                            "alpha": None,
                            "beta": None,
                            "value": int(val),
                            "note": "heuristic",
                        }
                    )
                    stack.pop()
                    if fr.incoming_move is not None:
                        board.undo_move(fr.incoming_move[0], fr.incoming_move[1])
                    if stack:
                        parent = stack[-1]
                        maximizing = parent.current_player == ai_player
                        if maximizing:
                            if val > parent.best_value:
                                parent.best_value = int(val)
                                parent.best_move = fr.incoming_move
                        else:
                            if val < parent.best_value:
                                parent.best_value = int(val)
                                parent.best_move = fr.incoming_move
                    continue

                if fr.i >= len(fr.moves):
                    emit(
                        {
                            "type": "exit",
                            "node_id": fr.node_id,
                            "parent_id": fr.parent_id,
                            "depth": depth - fr.depth_left,
                            "player": fr.current_player,
                            "move": fr.incoming_move,
                            "alpha": None,
                            "beta": None,
                            "value": int(fr.best_value if fr.best_value not in (-INF, INF) else 0),
                            "note": "return",
                        }
                    )
                    stack.pop()
                    if fr.incoming_move is not None:
                        board.undo_move(fr.incoming_move[0], fr.incoming_move[1])
                    if stack:
                        parent = stack[-1]
                        val = fr.best_value if fr.best_value not in (-INF, INF) else 0
                        maximizing = parent.current_player == ai_player
                        if maximizing:
                            if val > parent.best_value:
                                parent.best_value = int(val)
                                parent.best_move = fr.incoming_move
                        else:
                            if val < parent.best_value:
                                parent.best_value = int(val)
                                parent.best_move = fr.incoming_move
                    continue

                move = fr.moves[fr.i]
                fr.i += 1
                board.make_move(move[0], move[1], fr.current_player)
                child_id = new_id()
                child = Frame(
                    node_id=child_id,
                    parent_id=fr.node_id,
                    depth_left=fr.depth_left - 1,
                    current_player=3 - fr.current_player,
                    alpha=-INF,
                    beta=INF,
                    last_move=move,
                    moves=limited_moves(3 - fr.current_player),
                    i=0,
                    best_value=(INF if (3 - fr.current_player) != ai_player else -INF),
                    best_move=None,
                    incoming_move=move,
                )
                emit(
                    {
                        "type": "enter",
                        "node_id": child.node_id,
                        "parent_id": child.parent_id,
                        "depth": depth - child.depth_left,
                        "player": fr.current_player,
                        "move": move,
                        "alpha": None,
                        "beta": None,
                        "value": None,
                        "note": "expand",
                    }
                )
                stack.append(child)

            return root.best_move, int(root.best_value if root.best_value != -INF else 0)

        best_move, _v = minimax_dfs()
        return best_move, events

    def alphabeta_dfs() -> tuple[Move | None, int]:
        nonlocal node_id_seq
        INF = 10**18

        @dataclass
        class Frame:
            node_id: int
            parent_id: int | None
            depth_left: int
            current_player: int
            alpha: int
            beta: int
            last_move: Move | None
            moves: list[Move]
            i: int
            value: int
            best_move: Move | None
            incoming_move: Move | None
            maximizing: bool
            prunes: int

        def new_id() -> int:
            nonlocal node_id_seq
            node_id_seq += 1
            return node_id_seq

        root_id = new_id()
        root = Frame(
            node_id=root_id,
            parent_id=None,
            depth_left=depth,
            current_player=ai_player,
            alpha=-INF,
            beta=INF,
            last_move=board.last_move,
            moves=limited_moves(ai_player),
            i=0,
            value=-INF,
            best_move=None,
            incoming_move=None,
            maximizing=True,
            prunes=0,
        )
        emit(
            {
                "type": "enter",
                "node_id": root.node_id,
                "parent_id": None,
                "depth": depth - root.depth_left,
                "player": root.current_player,
                "move": None,
                "alpha": None,
                "beta": None,
                "value": None,
                "note": "root",
            }
        )

        stack: list[Frame] = [root]
        while stack:
            fr = stack[-1]

            term = terminal_value(fr.last_move)
            if term is not None:
                emit(
                    {
                        "type": "leaf",
                        "node_id": fr.node_id,
                        "parent_id": fr.parent_id,
                        "depth": depth - fr.depth_left,
                        "player": fr.current_player,
                        "move": fr.incoming_move,
                        "alpha": int(fr.alpha if fr.alpha != -INF else -1_000_000_000),
                        "beta": int(fr.beta if fr.beta != INF else 1_000_000_000),
                        "value": int(term),
                        "note": "terminal",
                    }
                )
                stack.pop()
                if fr.incoming_move is not None:
                    board.undo_move(fr.incoming_move[0], fr.incoming_move[1])
                if stack:
                    parent = stack[-1]
                    child_val = int(term)
                    if parent.maximizing:
                        if child_val > parent.value:
                            parent.value = child_val
                            parent.best_move = fr.incoming_move
                        parent.alpha = max(parent.alpha, parent.value)
                    else:
                        if child_val < parent.value:
                            parent.value = child_val
                            parent.best_move = fr.incoming_move
                        parent.beta = min(parent.beta, parent.value)
                    if parent.alpha >= parent.beta:
                        parent.prunes += 1
                        emit(
                            {
                                "type": "prune",
                                "node_id": parent.node_id,
                                "parent_id": parent.parent_id,
                                "depth": depth - parent.depth_left,
                                "player": parent.current_player,
                                "move": parent.incoming_move,
                                "alpha": int(parent.alpha if parent.alpha != -INF else -1_000_000_000),
                                "beta": int(parent.beta if parent.beta != INF else 1_000_000_000),
                                "value": int(parent.value if parent.value not in (-INF, INF) else 0),
                                "note": "cutoff",
                            }
                        )
                        parent.i = len(parent.moves)
                continue

            if fr.depth_left == 0:
                val = evaluate_board(board, ai_player)
                emit(
                    {
                        "type": "leaf",
                        "node_id": fr.node_id,
                        "parent_id": fr.parent_id,
                        "depth": depth - fr.depth_left,
                        "player": fr.current_player,
                        "move": fr.incoming_move,
                        "alpha": int(fr.alpha if fr.alpha != -INF else -1_000_000_000),
                        "beta": int(fr.beta if fr.beta != INF else 1_000_000_000),
                        "value": int(val),
                        "note": "heuristic",
                    }
                )
                stack.pop()
                if fr.incoming_move is not None:
                    board.undo_move(fr.incoming_move[0], fr.incoming_move[1])
                if stack:
                    parent = stack[-1]
                    if parent.maximizing:
                        if val > parent.value:
                            parent.value = int(val)
                            parent.best_move = fr.incoming_move
                        parent.alpha = max(parent.alpha, parent.value)
                    else:
                        if val < parent.value:
                            parent.value = int(val)
                            parent.best_move = fr.incoming_move
                        parent.beta = min(parent.beta, parent.value)
                    if parent.alpha >= parent.beta:
                        parent.prunes += 1
                        emit(
                            {
                                "type": "prune",
                                "node_id": parent.node_id,
                                "parent_id": parent.parent_id,
                                "depth": depth - parent.depth_left,
                                "player": parent.current_player,
                                "move": parent.incoming_move,
                                "alpha": int(parent.alpha if parent.alpha != -INF else -1_000_000_000),
                                "beta": int(parent.beta if parent.beta != INF else 1_000_000_000),
                                "value": int(parent.value if parent.value not in (-INF, INF) else 0),
                                "note": "cutoff",
                            }
                        )
                        parent.i = len(parent.moves)
                continue

            if fr.i >= len(fr.moves):
                emit(
                    {
                        "type": "exit",
                        "node_id": fr.node_id,
                        "parent_id": fr.parent_id,
                        "depth": depth - fr.depth_left,
                        "player": fr.current_player,
                        "move": fr.incoming_move,
                        "alpha": int(fr.alpha if fr.alpha != -INF else -1_000_000_000),
                        "beta": int(fr.beta if fr.beta != INF else 1_000_000_000),
                        "value": int(fr.value if fr.value not in (-INF, INF) else 0),
                        "note": "return",
                    }
                )
                stack.pop()
                if fr.incoming_move is not None:
                    board.undo_move(fr.incoming_move[0], fr.incoming_move[1])
                if stack:
                    parent = stack[-1]
                    child_val = fr.value if fr.value not in (-INF, INF) else 0
                    if parent.maximizing:
                        if child_val > parent.value:
                            parent.value = int(child_val)
                            parent.best_move = fr.incoming_move
                        parent.alpha = max(parent.alpha, parent.value)
                    else:
                        if child_val < parent.value:
                            parent.value = int(child_val)
                            parent.best_move = fr.incoming_move
                        parent.beta = min(parent.beta, parent.value)
                    if parent.alpha >= parent.beta:
                        parent.prunes += 1
                        emit(
                            {
                                "type": "prune",
                                "node_id": parent.node_id,
                                "parent_id": parent.parent_id,
                                "depth": depth - parent.depth_left,
                                "player": parent.current_player,
                                "move": parent.incoming_move,
                                "alpha": int(parent.alpha if parent.alpha != -INF else -1_000_000_000),
                                "beta": int(parent.beta if parent.beta != INF else 1_000_000_000),
                                "value": int(parent.value if parent.value not in (-INF, INF) else 0),
                                "note": "cutoff",
                            }
                        )
                        parent.i = len(parent.moves)
                continue

            move = fr.moves[fr.i]
            fr.i += 1
            board.make_move(move[0], move[1], fr.current_player)
            child_id = new_id()
            child_player = 3 - fr.current_player
            maximizing = child_player == ai_player
            child = Frame(
                node_id=child_id,
                parent_id=fr.node_id,
                depth_left=fr.depth_left - 1,
                current_player=child_player,
                alpha=fr.alpha,
                beta=fr.beta,
                last_move=move,
                moves=limited_moves(child_player),
                i=0,
                value=(-INF if maximizing else INF),
                best_move=None,
                incoming_move=move,
                maximizing=maximizing,
                prunes=0,
            )
            emit(
                {
                    "type": "enter",
                    "node_id": child.node_id,
                    "parent_id": child.parent_id,
                    "depth": depth - child.depth_left,
                    "player": fr.current_player,
                    "move": move,
                    "alpha": int(fr.alpha if fr.alpha != -INF else -1_000_000_000),
                    "beta": int(fr.beta if fr.beta != INF else 1_000_000_000),
                    "value": None,
                    "note": "expand",
                }
            )
            stack.append(child)

        return root.best_move, int(root.value if root.value != -INF else 0)

    best_move, _v = alphabeta_dfs()
    return best_move, events


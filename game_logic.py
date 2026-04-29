from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Move = tuple[int, int]


@dataclass(frozen=True)
class WinnerResult:
    player: int
    line: list[Move]


class CaroBoard:
    def __init__(self, size: int = 15, win_length: int = 5) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if win_length <= 1 or win_length > size:
            raise ValueError("win_length must be in [2, size]")

        self.size = size
        self.win_length = win_length
        self.board: np.ndarray = np.zeros((size, size), dtype=np.int8)
        self.moves_played: list[tuple[int, int, int]] = []
        self.last_move: Move | None = None

    def reset(self) -> None:
        self.board.fill(0)
        self.moves_played.clear()
        self.last_move = None

    def is_valid_move(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row, col] == 0

    def make_move(self, row: int, col: int, player: int) -> bool:
        if player not in (1, 2):
            raise ValueError("player must be 1 or 2")
        if not self.is_valid_move(row, col):
            return False

        self.board[row, col] = player
        self.moves_played.append((row, col, player))
        self.last_move = (row, col)
        return True

    def undo_move(self, row: int, col: int) -> None:
        if not (0 <= row < self.size and 0 <= col < self.size):
            return
        if self.board[row, col] == 0:
            return

        self.board[row, col] = 0
        for i in range(len(self.moves_played) - 1, -1, -1):
            r, c, _p = self.moves_played[i]
            if r == row and c == col:
                self.moves_played.pop(i)
                break
        self.last_move = (self.moves_played[-1][0], self.moves_played[-1][1]) if self.moves_played else None

    def is_full(self) -> bool:
        return not np.any(self.board == 0)

    def check_winner(self, row: int, col: int) -> WinnerResult | None:
        player = int(self.board[row, col])
        if player == 0:
            return None

        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for dr, dc in directions:
            line: list[Move] = [(row, col)]

            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and int(self.board[r, c]) == player:
                line.insert(0, (r, c))
                r -= dr
                c -= dc

            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and int(self.board[r, c]) == player:
                line.append((r, c))
                r += dr
                c += dc

            if len(line) >= self.win_length:
                return WinnerResult(player=player, line=line)

        return None

    def get_possible_moves(self, radius: int = 2) -> list[Move]:
        if radius < 0:
            raise ValueError("radius must be >= 0")

        occupied = np.argwhere(self.board != 0)
        if occupied.size == 0:
            center = self.size // 2
            return [(center, center)]

        candidates: set[Move] = set()
        for r, c in occupied:
            r0 = max(0, int(r) - radius)
            r1 = min(self.size - 1, int(r) + radius)
            c0 = max(0, int(c) - radius)
            c1 = min(self.size - 1, int(c) + radius)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if self.board[rr, cc] == 0:
                        candidates.add((rr, cc))

        return sorted(candidates)


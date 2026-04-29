from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from ai_algorithms import AlphaBetaAI
from game_logic import CaroBoard
from gui import MainWindow


def test_game_logic() -> None:
    board = CaroBoard(size=15)
    row = 7
    for col in range(3, 8):
        ok = board.make_move(row, col, 1)
        assert ok
    win = board.check_winner(row, 7)
    assert win is not None
    assert win.player == 1

    board.reset()
    board.make_move(7, 7, 1)
    moves = board.get_possible_moves(radius=2)
    assert (7, 7) not in moves
    assert len(moves) == 24
    assert all(5 <= r <= 9 and 5 <= c <= 9 for r, c in moves)


def test_ai_move() -> None:
    board = CaroBoard(size=15)
    board.make_move(7, 7, 1)
    board.make_move(7, 8, 2)
    board.make_move(8, 8, 1)
    ai = AlphaBetaAI(depth=2)
    move, stats = ai.choose_move(board, ai_player=2)
    assert move is not None
    r, c = move
    assert board.is_valid_move(r, c)
    assert stats.time_taken_ms >= 0.0


def test_gui_smoke() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    board = CaroBoard(size=15)
    win = MainWindow(board)
    win.show()
    QTimer.singleShot(150, win.close)
    QTimer.singleShot(200, app.quit)
    app.exec()


def main() -> None:
    np.random.seed(0)
    test_game_logic()
    test_ai_move()
    test_gui_smoke()
    print("smoke_ok")


if __name__ == "__main__":
    main()


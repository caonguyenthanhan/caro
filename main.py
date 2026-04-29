from __future__ import annotations

import sys

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QMessageBox

from ai_algorithms import AlphaBetaAI, MinimaxAI, RandomAI, evaluate_board, iter_search_demo
from game_logic import CaroBoard, Move
from gui import MainWindow


class AiWorker(QObject):
    finished = pyqtSignal(int, int, float, int, int)
    demo_ready = pyqtSignal(object, object)

    @pyqtSlot(str, int, object)
    def compute(self, mode: str, depth: int, board_state: np.ndarray) -> None:
        board = CaroBoard(size=int(board_state.shape[0]))
        board.board[:, :] = board_state

        if mode == "AI Random":
            ai = RandomAI()
        elif mode == "AI Minimax":
            ai = MinimaxAI(depth=depth)
        else:
            ai = AlphaBetaAI(depth=depth)

        move, stats = ai.choose_move(board, ai_player=2)
        if move is None:
            self.finished.emit(-1, -1, stats.time_taken_ms, stats.nodes_evaluated, stats.heuristic_score)
            return
        r, c = move
        self.finished.emit(int(r), int(c), stats.time_taken_ms, stats.nodes_evaluated, stats.heuristic_score)

    @pyqtSlot(str, int, int, object)
    def compute_demo(self, algorithm: str, depth: int, max_children: int, board_state: np.ndarray) -> None:
        board = CaroBoard(size=int(board_state.shape[0]))
        board.board[:, :] = board_state
        best_move, events = iter_search_demo(
            board=board,
            algorithm=algorithm,
            depth=depth,
            ai_player=2,
            max_children=max_children,
        )
        self.demo_ready.emit(best_move, events)


class GameController(QObject):
    request_ai = pyqtSignal(str, int, object)
    request_demo = pyqtSignal(str, int, int, object)

    def __init__(self) -> None:
        super().__init__()
        self.board = CaroBoard(size=15)
        self.window = MainWindow(self.board)
        self.mode: str = "AI Alpha-Beta"
        self.depth: int = 3
        self.current_player = 1
        self.game_over = False
        self.thinking = False

        self.window.cell_clicked.connect(self.on_cell_clicked)
        self.window.new_game_clicked.connect(self.new_game)
        self.window.undo_clicked.connect(self.undo)
        self.window.selection_changed.connect(self.on_selection_changed)
        self.window.demo_start_clicked.connect(self.start_demo)
        self.window.demo_pause_clicked.connect(self.toggle_demo_pause)
        self.window.demo_step_clicked.connect(self.demo_step)
        self.window.demo_jump_clicked.connect(self.demo_jump)
        self.window.demo_clear_clicked.connect(self.clear_demo)

        self.thread = QThread()
        self.worker = AiWorker()
        self.worker.moveToThread(self.thread)
        self.request_ai.connect(self.worker.compute)
        self.request_demo.connect(self.worker.compute_demo)
        self.worker.finished.connect(self.on_ai_finished)
        self.worker.demo_ready.connect(self.on_demo_ready)
        self.thread.start()

        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self._demo_tick)
        self.demo_events: list[dict] = []
        self.demo_index = 0
        self.demo_items_by_id: dict[int, object] = {}
        self.demo_preparing = False
        self.demo_autoplay = False

        self._refresh_stats(None, None)
        self.window.show()

    def shutdown(self) -> None:
        self.thread.quit()
        self.thread.wait(1500)

    def on_selection_changed(self, mode: str, depth: int) -> None:
        self.mode = mode
        self.depth = depth
        self._refresh_stats(None, None)

    def new_game(self) -> None:
        if self.thinking:
            return
        self.board.reset()
        self.current_player = 1
        self.game_over = False
        self.window.set_thinking(False)
        self.window.board_widget.update()
        self._refresh_stats(None, None)

        self.clear_demo()

    def undo(self) -> None:
        if self.thinking:
            return
        if not self.board.moves_played:
            return

        last_player = self.board.moves_played[-1][2]
        last_r, last_c = self.board.moves_played[-1][0], self.board.moves_played[-1][1]
        self.board.undo_move(last_r, last_c)

        if self.mode != "Người vs Người" and last_player == 2 and self.board.moves_played:
            prev_player = self.board.moves_played[-1][2]
            if prev_player == 1:
                prev_r, prev_c = self.board.moves_played[-1][0], self.board.moves_played[-1][1]
                self.board.undo_move(prev_r, prev_c)
                self.current_player = 1
            else:
                self.current_player = 2
        else:
            self.current_player = int(last_player)

        self.game_over = False
        self.window.board_widget.update()
        self._refresh_stats(None, None)

        self.clear_demo()

    def on_cell_clicked(self, row: int, col: int) -> None:
        if self.game_over or self.thinking:
            return

        if self.current_player == 2 and self.mode != "Người vs Người":
            return

        if not self.board.make_move(row, col, self.current_player):
            return

        self.window.board_widget.update()
        if self._handle_end_if_any((row, col)):
            return

        self.current_player = 3 - self.current_player
        self._refresh_stats(None, None)

        if self.current_player == 2 and self.mode != "Người vs Người":
            self._start_ai_turn()

    def _start_ai_turn(self) -> None:
        if self.thinking or self.game_over:
            return
        self.thinking = True
        self.window.set_thinking(True)
        board_state = self.board.board.copy()
        self.request_ai.emit(self.mode, int(self.depth), board_state)

    def _demo_speed_ms(self, label: str) -> int:
        if label == "Chậm":
            return 450
        if label == "Vừa":
            return 140
        if label == "Nhanh":
            return 40
        if label == "Siêu tốc":
            return 10
        return 0

    def start_demo(self) -> None:
        if self.demo_preparing:
            return
        if self.demo_timer.isActive():
            self.demo_timer.stop()

        algo_label, depth, topn, speed_label = self.window.demo_settings()
        algorithm = "Alpha-Beta" if algo_label == "Alpha-Beta" else "Minimax"

        self.demo_preparing = True
        self.window.set_demo_status("Demo: đang tạo nhánh suy nghĩ...")
        self.window.set_demo_running(True)

        board_state = self.board.board.copy()
        self.request_demo.emit(algorithm, int(depth), int(topn), board_state)

        self.demo_autoplay = speed_label != "Từng bước"

    def on_demo_ready(self, best_move: object, events: object) -> None:
        self.demo_preparing = False
        self.demo_events = list(events)
        self.demo_index = 0
        self.demo_items_by_id = {}
        self.window.clear_demo()
        self.window.board_widget.set_demo_focus(None)

        algo_label, depth, topn, speed_label = self.window.demo_settings()
        info = f"Demo: {algo_label}, depth={depth}, topN={topn}, events={len(self.demo_events)}"
        if best_move is not None:
            info += f", best~={best_move}"
        self.window.set_demo_status(info)

        if self.demo_autoplay and self.demo_events:
            interval = self._demo_speed_ms(speed_label)
            self.demo_timer.start(max(1, interval))
        else:
            self.window.set_demo_running(False)

    def toggle_demo_pause(self) -> None:
        if self.demo_preparing:
            return
        if not self.demo_events:
            return
        if self.demo_timer.isActive():
            self.demo_timer.stop()
            self.window.set_demo_status("Demo: pause")
            self.window.set_demo_running(False)
        else:
            algo_label, depth, topn, speed_label = self.window.demo_settings()
            if speed_label == "Từng bước":
                return
            interval = self._demo_speed_ms(speed_label)
            self.demo_timer.start(max(1, interval))
            self.window.set_demo_running(True)
            self.window.set_demo_status("Demo: chạy")

    def demo_step(self) -> None:
        if self.demo_preparing:
            return
        if self.demo_timer.isActive():
            return
        self._demo_tick()

    def demo_jump(self) -> None:
        if self.demo_preparing:
            return
        if self.demo_timer.isActive():
            return
        for _ in range(50):
            if self.demo_index >= len(self.demo_events):
                break
            self._demo_tick()

    def clear_demo(self) -> None:
        if self.demo_timer.isActive():
            self.demo_timer.stop()
        self.demo_preparing = False
        self.demo_events = []
        self.demo_index = 0
        self.demo_items_by_id = {}
        self.window.clear_demo()
        self.window.board_widget.set_demo_focus(None)
        self.window.set_demo_running(False)

    def _demo_tick(self) -> None:
        if self.demo_index >= len(self.demo_events):
            if self.demo_timer.isActive():
                self.demo_timer.stop()
            self.window.set_demo_status("Demo: hoàn tất")
            self.window.set_demo_running(False)
            return

        ev = self.demo_events[self.demo_index]
        self.demo_index += 1

        move = ev.get("move")
        if isinstance(move, tuple) and len(move) == 2:
            self.window.board_widget.set_demo_focus((int(move[0]), int(move[1])))
        else:
            self.window.board_widget.set_demo_focus(None)

        node_id = int(ev.get("node_id", 0))
        parent_id = ev.get("parent_id")
        parent_id_int = None if parent_id is None else int(parent_id)

        depth = int(ev.get("depth", 0))
        player = int(ev.get("player", 2))
        alpha = ev.get("alpha")
        beta = ev.get("beta")
        value = ev.get("value")
        note = str(ev.get("note", ev.get("type", "")))

        self.window.upsert_demo_node(
            items_by_id=self.demo_items_by_id,
            node_id=node_id,
            parent_id=parent_id_int,
            move=move if isinstance(move, tuple) else None,
            depth=depth,
            player=player,
            alpha=None if alpha is None else int(alpha),
            beta=None if beta is None else int(beta),
            value=None if value is None else int(value),
            note=note,
        )

        etype = ev.get("type")
        if etype in ("enter", "leaf", "exit", "prune", "update"):
            self.window.append_demo_log(
                f"#{node_id} {etype} d={depth} p={'X' if player==1 else 'O'} move={move} a={alpha} b={beta} v={value} {note}"
            )

        if self.demo_events:
            self.window.set_demo_status(f"Demo: {self.demo_index}/{len(self.demo_events)}")

    def on_ai_finished(self, row: int, col: int, time_ms: float, nodes: int, score: int) -> None:
        self.thinking = False
        self.window.set_thinking(False)

        if self.game_over:
            return
        if row < 0 or col < 0:
            self._refresh_stats(time_ms, nodes)
            return

        if not self.board.make_move(int(row), int(col), 2):
            self._refresh_stats(time_ms, nodes)
            return

        self.window.board_widget.update()
        if self._handle_end_if_any((int(row), int(col))):
            self._refresh_stats(time_ms, nodes)
            return

        self.current_player = 1
        self._refresh_stats(time_ms, nodes)

    def _handle_end_if_any(self, last_move: Move) -> bool:
        win = self.board.check_winner(last_move[0], last_move[1])
        if win is not None:
            self.game_over = True
            self.window.set_thinking(True)
            who = "Người chơi (X)" if win.player == 1 else "AI (O)"
            QMessageBox.information(self.window, "Kết thúc", f"{who} thắng!")
            self.window.set_thinking(False)
            return True
        if self.board.is_full():
            self.game_over = True
            QMessageBox.information(self.window, "Kết thúc", "Hòa!")
            return True
        return False

    def _refresh_stats(self, time_ms: float | None, nodes: int | None) -> None:
        algo = self.mode
        score = evaluate_board(self.board, ai_player=2)
        self.window.update_stats(algo, time_ms, nodes, score)


def main() -> int:
    app = QApplication(sys.argv)
    controller = GameController()
    exit_code = app.exec()
    controller.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


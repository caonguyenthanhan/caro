from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from game_logic import CaroBoard, Move


@dataclass(frozen=True)
class UiSelection:
    mode: str
    depth: int


class BoardWidget(QWidget):
    cell_clicked = pyqtSignal(int, int)

    def __init__(self, board: CaroBoard, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._board = board
        self._margin = 18
        self._cell = 34
        self._grid_pen = QPen(QColor("#2A2F3A"), 1)
        self._highlight_pen = QPen(QColor("#F5C542"), 2)
        self._demo_pen = QPen(QColor("#22D3EE"), 2)
        self._locked = False
        self._demo_focus: Move | None = None

        self.setMouseTracking(True)
        min_w, min_h = self._preferred_size()
        self.setMinimumSize(min_w, min_h)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())

    def _preferred_size(self) -> tuple[int, int]:
        side = self._margin * 2 + self._cell * (self._board.size - 1)
        return side, side

    def lock_input(self, locked: bool) -> None:
        self._locked = locked

    def set_demo_focus(self, move: Move | None) -> None:
        self._demo_focus = move
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.fillRect(self.rect(), QColor("#0E1117"))

        painter.setPen(self._grid_pen)
        size = self._board.size
        left = self._margin
        top = self._margin
        right = left + self._cell * (size - 1)
        bottom = top + self._cell * (size - 1)

        for i in range(size):
            x = left + i * self._cell
            painter.drawLine(x, top, x, bottom)
            y = top + i * self._cell
            painter.drawLine(left, y, right, y)

        last = self._board.last_move
        if last is not None:
            r, c = last
            cx = left + c * self._cell
            cy = top + r * self._cell
            painter.setPen(self._highlight_pen)
            painter.drawRect(cx - self._cell // 2 + 1, cy - self._cell // 2 + 1, self._cell - 2, self._cell - 2)

        if self._demo_focus is not None:
            r, c = self._demo_focus
            cx = left + c * self._cell
            cy = top + r * self._cell
            painter.setPen(self._demo_pen)
            painter.drawRect(cx - self._cell // 2 + 3, cy - self._cell // 2 + 3, self._cell - 6, self._cell - 6)

        for r in range(size):
            for c in range(size):
                v = int(self._board.board[r, c])
                if v == 0:
                    continue
                cx = left + c * self._cell
                cy = top + r * self._cell
                self._draw_piece(painter, cx, cy, v)

    def _draw_piece(self, painter: QPainter, cx: int, cy: int, player: int) -> None:
        radius = self._cell // 2 - 3
        rect = (cx - radius, cy - radius, radius * 2, radius * 2)

        gradient = QLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius)
        if player == 1:
            gradient.setColorAt(0.0, QColor("#3B82F6"))
            gradient.setColorAt(1.0, QColor("#60A5FA"))
        else:
            gradient.setColorAt(0.0, QColor("#EF4444"))
            gradient.setColorAt(1.0, QColor("#F87171"))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(*rect)

        inner = radius - 4
        painter.setBrush(QColor(255, 255, 255, 28))
        painter.drawEllipse(cx - inner, cy - inner, inner * 2, inner * 2)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._locked:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        row_col = self._pos_to_cell(event.position().x(), event.position().y())
        if row_col is None:
            return
        r, c = row_col
        self.cell_clicked.emit(r, c)

    def _pos_to_cell(self, x: float, y: float) -> Move | None:
        left = self._margin
        top = self._margin
        size = self._board.size
        right = left + self._cell * (size - 1)
        bottom = top + self._cell * (size - 1)
        if x < left - self._cell / 2 or x > right + self._cell / 2:
            return None
        if y < top - self._cell / 2 or y > bottom + self._cell / 2:
            return None

        col = int(round((x - left) / self._cell))
        row = int(round((y - top) / self._cell))
        if 0 <= row < size and 0 <= col < size:
            return row, col
        return None


class MainWindow(QMainWindow):
    new_game_clicked = pyqtSignal()
    undo_clicked = pyqtSignal()
    selection_changed = pyqtSignal(str, int)
    cell_clicked = pyqtSignal(int, int)

    demo_start_clicked = pyqtSignal()
    demo_pause_clicked = pyqtSignal()
    demo_step_clicked = pyqtSignal()
    demo_jump_clicked = pyqtSignal()
    demo_clear_clicked = pyqtSignal()

    def __init__(self, board: CaroBoard) -> None:
        super().__init__()
        self.setWindowTitle("AI Caro Gomoku - Đồ án Trí tuệ nhân tạo")
        self._board = board

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        self.board_widget = BoardWidget(board)
        self.board_widget.cell_clicked.connect(self.cell_clicked)
        board_frame = QFrame()
        board_frame.setObjectName("BoardFrame")
        board_layout = QVBoxLayout(board_frame)
        board_layout.setContentsMargins(14, 14, 14, 14)
        board_layout.addWidget(self.board_widget)
        layout.addWidget(board_frame, 2)

        panel = QFrame()
        panel.setObjectName("ControlPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 16, 16, 16)
        panel_layout.setSpacing(12)

        title = QLabel("AI Caro Gomoku - Đồ án Trí tuệ nhân tạo")
        title.setObjectName("Title")
        title.setWordWrap(True)
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        panel_layout.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("Tabs")
        panel_layout.addWidget(self.tabs)

        play_tab = QWidget()
        play_layout = QVBoxLayout(play_tab)
        play_layout.setContentsMargins(0, 0, 0, 0)
        play_layout.setSpacing(12)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Người vs Người", "AI Random", "AI Minimax", "AI Alpha-Beta"])
        self.mode_combo.currentTextChanged.connect(self._emit_selection)
        play_layout.addWidget(self._labeled("Thuật toán (Player 2)", self.mode_combo))

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 5)
        self.depth_spin.setValue(3)
        self.depth_spin.valueChanged.connect(self._emit_selection)
        play_layout.addWidget(self._labeled("Depth", self.depth_spin))

        self.new_game_btn = QPushButton("New Game")
        self.undo_btn = QPushButton("Undo Move")
        self.new_game_btn.clicked.connect(self.new_game_clicked)
        self.undo_btn.clicked.connect(self.undo_clicked)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addWidget(self.new_game_btn)
        btn_row.addWidget(self.undo_btn)
        play_layout.addLayout(btn_row)

        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setObjectName("Status")
        self.status_label.setWordWrap(True)
        play_layout.addWidget(self.status_label)

        stats = QGroupBox("Statistics")
        stats.setObjectName("Stats")
        stats_layout = QVBoxLayout(stats)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.setSpacing(8)

        self.stat_algo = QLabel("Thuật toán: -")
        self.stat_time = QLabel("Thời gian: -")
        self.stat_nodes = QLabel("Nodes: -")
        self.stat_score = QLabel("Heuristic: -")
        for w in (self.stat_algo, self.stat_time, self.stat_nodes, self.stat_score):
            w.setObjectName("StatLine")
            stats_layout.addWidget(w)

        play_layout.addWidget(stats)
        play_layout.addStretch(1)
        self.tabs.addTab(play_tab, "Chơi")

        demo_tab = QWidget()
        demo_layout = QVBoxLayout(demo_tab)
        demo_layout.setContentsMargins(0, 0, 0, 0)
        demo_layout.setSpacing(10)

        self.demo_algo = QComboBox()
        self.demo_algo.addItems(["Alpha-Beta", "Minimax"])
        demo_layout.addWidget(self._labeled("Thuật toán (Demo)", self.demo_algo))

        self.demo_depth = QSpinBox()
        self.demo_depth.setRange(1, 5)
        self.demo_depth.setValue(3)
        demo_layout.addWidget(self._labeled("Depth (Demo)", self.demo_depth))

        self.demo_topn = QSpinBox()
        self.demo_topn.setRange(4, 30)
        self.demo_topn.setValue(12)
        demo_layout.addWidget(self._labeled("Giới hạn nhánh / node", self.demo_topn))

        self.demo_speed = QComboBox()
        self.demo_speed.addItems(["Từng bước", "Chậm", "Vừa", "Nhanh", "Siêu tốc"])
        self.demo_speed.setCurrentText("Vừa")
        demo_layout.addWidget(self._labeled("Tốc độ", self.demo_speed))

        self.demo_autoscroll = QCheckBox("Auto-scroll log")
        self.demo_autoscroll.setChecked(True)
        self.demo_autoscroll.setStyleSheet("color: #C9D1D9;")
        demo_layout.addWidget(self.demo_autoscroll)

        self.demo_start_btn = QPushButton("Run Demo")
        self.demo_pause_btn = QPushButton("Pause")
        self.demo_step_btn = QPushButton("Step")
        self.demo_jump_btn = QPushButton(">>")
        self.demo_clear_btn = QPushButton("Clear")
        self.demo_start_btn.clicked.connect(self.demo_start_clicked)
        self.demo_pause_btn.clicked.connect(self.demo_pause_clicked)
        self.demo_step_btn.clicked.connect(self.demo_step_clicked)
        self.demo_jump_btn.clicked.connect(self.demo_jump_clicked)
        self.demo_clear_btn.clicked.connect(self.demo_clear_clicked)

        demo_btn_row = QHBoxLayout()
        demo_btn_row.setSpacing(10)
        demo_btn_row.addWidget(self.demo_start_btn)
        demo_btn_row.addWidget(self.demo_pause_btn)
        demo_btn_row.addWidget(self.demo_step_btn)
        demo_btn_row.addWidget(self.demo_jump_btn)
        demo_btn_row.addWidget(self.demo_clear_btn)
        demo_layout.addLayout(demo_btn_row)

        self.demo_status = QLabel("Demo: -")
        self.demo_status.setObjectName("Status")
        self.demo_status.setWordWrap(True)
        demo_layout.addWidget(self.demo_status)

        self.demo_tree = QTreeWidget()
        self.demo_tree.setObjectName("DemoTree")
        self.demo_tree.setHeaderLabels(["Node", "Move", "D", "P", "α", "β", "Val", "Note"])
        self.demo_tree.setAlternatingRowColors(True)
        self.demo_tree.setColumnWidth(0, 56)
        self.demo_tree.setColumnWidth(1, 70)
        self.demo_tree.setColumnWidth(2, 28)
        self.demo_tree.setColumnWidth(3, 24)
        self.demo_tree.setColumnWidth(4, 52)
        self.demo_tree.setColumnWidth(5, 52)
        self.demo_tree.setColumnWidth(6, 60)
        demo_layout.addWidget(self.demo_tree, 2)

        self.demo_log = QPlainTextEdit()
        self.demo_log.setReadOnly(True)
        self.demo_log.setObjectName("DemoLog")
        demo_layout.addWidget(self.demo_log, 1)

        self.tabs.addTab(demo_tab, "Demo")
        layout.addWidget(panel, 1)

        self.setStyleSheet(
            """
            QMainWindow { background: #0B0E14; }
            QFrame#BoardFrame, QFrame#ControlPanel {
                background: #0E1117;
                border: 1px solid #232A34;
                border-radius: 14px;
            }
            QLabel#Title { color: #E6EDF3; }
            QLabel#Status { color: #AAB2BF; }
            QGroupBox#Stats {
                color: #E6EDF3;
                border: 1px solid #232A34;
                border-radius: 12px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox#Stats::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QLabel#StatLine { color: #C9D1D9; }
            QComboBox, QSpinBox {
                background: #0B1220;
                color: #E6EDF3;
                border: 1px solid #243041;
                border-radius: 10px;
                padding: 7px 10px;
            }
            QTabWidget#Tabs::pane { border: 0px; }
            QTabBar::tab {
                background: #0B1220;
                color: #AAB2BF;
                padding: 8px 10px;
                border: 1px solid #243041;
                border-bottom: 0px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 6px;
            }
            QTabBar::tab:selected { background: #0E1117; color: #E6EDF3; }
            QPushButton {
                background: #1F6FEB;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 9px 12px;
            }
            QPushButton:hover { background: #2B7BFF; }
            QPushButton:pressed { background: #1A5BD1; }
            QPushButton:disabled { background: #2A2F3A; color: #9AA3AE; }
            QTreeWidget#DemoTree, QPlainTextEdit#DemoLog {
                background: #0B1220;
                color: #E6EDF3;
                border: 1px solid #243041;
                border-radius: 12px;
                padding: 6px;
            }
            """
        )

        self._emit_selection()

    def _labeled(self, text: str, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #AAB2BF;")
        lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        layout.addWidget(lbl)
        layout.addWidget(widget)
        return box

    def _emit_selection(self) -> None:
        self.selection_changed.emit(self.mode_combo.currentText(), int(self.depth_spin.value()))

    def set_thinking(self, thinking: bool) -> None:
        self.board_widget.lock_input(thinking)
        self.mode_combo.setEnabled(not thinking)
        self.depth_spin.setEnabled(not thinking)
        self.new_game_btn.setEnabled(not thinking)
        self.undo_btn.setEnabled(not thinking)
        self.status_label.setText("AI đang suy nghĩ..." if thinking else "Sẵn sàng")

    def set_demo_running(self, running: bool) -> None:
        self.demo_start_btn.setEnabled(not running)
        self.demo_algo.setEnabled(not running)
        self.demo_depth.setEnabled(not running)
        self.demo_topn.setEnabled(not running)
        self.demo_speed.setEnabled(not running)
        self.demo_pause_btn.setEnabled(running)
        self.demo_step_btn.setEnabled(True)
        self.demo_jump_btn.setEnabled(True)

    def clear_demo(self) -> None:
        self.demo_tree.clear()
        self.demo_log.clear()
        self.demo_status.setText("Demo: -")

    def append_demo_log(self, text: str) -> None:
        self.demo_log.appendPlainText(text)
        if self.demo_autoscroll.isChecked():
            bar = self.demo_log.verticalScrollBar()
            bar.setValue(bar.maximum())

    def set_demo_status(self, text: str) -> None:
        self.demo_status.setText(text)

    def demo_settings(self) -> tuple[str, int, int, str]:
        return (
            self.demo_algo.currentText(),
            int(self.demo_depth.value()),
            int(self.demo_topn.value()),
            self.demo_speed.currentText(),
        )

    def upsert_demo_node(
        self,
        items_by_id: dict[int, QTreeWidgetItem],
        node_id: int,
        parent_id: int | None,
        move: Move | None,
        depth: int,
        player: int,
        alpha: int | None,
        beta: int | None,
        value: int | None,
        note: str,
    ) -> None:
        item = items_by_id.get(node_id)
        if item is None:
            if parent_id is None:
                item = QTreeWidgetItem(self.demo_tree)
            else:
                parent = items_by_id.get(parent_id)
                item = QTreeWidgetItem(parent if parent is not None else self.demo_tree)
            items_by_id[node_id] = item

        item.setText(0, str(node_id))
        item.setText(1, "-" if move is None else f"({move[0]},{move[1]})")
        item.setText(2, str(depth))
        item.setText(3, "X" if player == 1 else "O")
        item.setText(4, "-" if alpha is None else str(alpha))
        item.setText(5, "-" if beta is None else str(beta))
        item.setText(6, "-" if value is None else str(value))
        item.setText(7, note)

        if note in ("cutoff", "prune"):
            bg = QBrush(QColor("#3B1D1D"))
            for col in range(8):
                item.setBackground(col, bg)
        elif note in ("best↑", "best↓"):
            bg = QBrush(QColor("#0B2A3A"))
            for col in range(8):
                item.setBackground(col, bg)
        self.demo_tree.expandToDepth(2)

    def update_stats(self, algorithm: str, time_ms: float | None, nodes: int | None, score: int | None) -> None:
        self.stat_algo.setText(f"Thuật toán: {algorithm}")
        self.stat_time.setText("Thời gian: -" if time_ms is None else f"Thời gian: {time_ms:.2f} ms")
        self.stat_nodes.setText("Nodes: -" if nodes is None else f"Nodes: {nodes}")
        self.stat_score.setText("Heuristic: -" if score is None else f"Heuristic: {score}")


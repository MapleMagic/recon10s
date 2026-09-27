#!/usr/bin/env python3
"""
recon10s_dropwin — the window a dropsonde dot opens.

A separate top-level window per click, so several drops can be compared
side by side. The figure is recon10s_skewt.render; this adds stepping to the
previous/next drop by number, the raw TEMP DROP message (to check the decode
against), and saving to PNG through the standard toolbar.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

import recon10s_skewt as skewt


class DropsondeWindow(QWidget):
    def __init__(self, drops: List, index: int, sat_for: Callable, centre_for: Callable,
                 theme: str = "dark", parent=None, on_show: Optional[Callable] = None):
        """
        drops      the loaded set, ordered by number
        index      which one to show first
        sat_for    drop -> GoesImage (or None) for the inset
        centre_for drop -> (lat, lon) storm centre, or None
        """
        super().__init__(parent, Qt.WindowType.Window)
        self.drops, self.index = drops, index
        self.sat_for, self.centre_for, self.theme = sat_for, centre_for, theme
        self.on_show = on_show
        self.resize(1200, 930)

        self.figure = Figure(figsize=(12, 9), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.btn_prev = QPushButton("\u25c0 Previous drop")
        self.btn_next = QPushButton("Next drop \u25b6")
        self.btn_prev.clicked.connect(lambda: self.show_index(self.index - 1))
        self.btn_next.clicked.connect(lambda: self.show_index(self.index + 1))
        self.btn_raw = QPushButton("Raw message")
        self.btn_raw.setCheckable(True)
        self.btn_raw.toggled.connect(lambda on: self.raw.setVisible(on))
        self.position = QLabel()
        self.position.setObjectName("hint")

        bar = QHBoxLayout()
        bar.addWidget(self.btn_prev)
        bar.addWidget(self.btn_next)
        bar.addWidget(self.position)
        bar.addStretch(1)
        bar.addWidget(self.btn_raw)
        bar.addWidget(self.toolbar)

        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setVisible(False)
        self.raw.setMaximumHeight(200)
        self.raw.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(bar)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.canvas)
        split.addWidget(self.raw)
        layout.addWidget(split, 1)

        self.show_index(index)

    def show_index(self, index: int):
        if not self.drops:
            return
        self.index = max(0, min(len(self.drops) - 1, index))
        drop = self.drops[self.index]
        skewt.render(self.figure, drop, sat_img=self.sat_for(drop),
                     centre=self.centre_for(drop), theme=self.theme)
        self.canvas.draw_idle()
        when = f"{drop.release_time:%H:%MZ}" if drop.release_time else ""
        self.setWindowTitle(f"Dropsonde #{drop.number} \u2014 {drop.aircraft} {when} "
                            f"{drop.location}".strip())
        self.position.setText(f"  {self.index + 1} of {len(self.drops)}")
        self.btn_prev.setEnabled(self.index > 0)
        self.btn_next.setEnabled(self.index < len(self.drops) - 1)
        self.raw.setPlainText(f"{drop.source}\n\n{drop.raw}")
        if self.on_show is not None:
            self.on_show(drop)   # lets the owner fetch this drop's satellite frame

#!/usr/bin/env python3
"""
recon10s_motion — the storm-motion picker shared by the radar tabs, for
storm-relative velocity.

Sources, best first: the recon centre fixes (a straight-line fit of the
last six hours), NHC's advisory motion (CurrentStorms.json, converted from
mph), the working best track (NHC's b-deck, or JTWC's for the West Pacific),
or typed in. A source that is not available falls back to the next.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QSpinBox

SOURCES = [("Recon fixes", "fixes"), ("NHC advisory", "nhc"), ("Best track (last 12 h)", "track"),
           ("Typed in", "manual")]


class MotionBox(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, title: str = "Storm motion", sources=SOURCES, parent=None):
        super().__init__(title, parent)
        self._known: Dict[str, Tuple[Tuple[float, float], str]] = {}
        f = QFormLayout(self)
        self.source = QComboBox()
        for label, key in sources:
            self.source.addItem(label, key)
        self.source.currentIndexChanged.connect(self._refresh)
        f.addRow("From", self.source)
        row = QHBoxLayout()
        self.heading = QSpinBox()
        self.heading.setRange(0, 359)
        self.heading.setWrapping(True)
        self.heading.setSuffix("°")
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.0, 70.0)
        self.speed.setDecimals(1)
        self.speed.setSuffix(" kt")
        for w in (self.heading, self.speed):
            w.valueChanged.connect(self._typed)
            row.addWidget(w)
        f.addRow("Toward", row)
        self.note = QLabel("")
        self.note.setObjectName("hint")
        self.note.setWordWrap(True)
        f.addRow(self.note)
        self._refresh()

    def set_source(self, key: str, motion: Optional[Tuple[float, float]], note: str = ""):
        """Make a source available (motion is (heading toward, kt)) or remove it (None)."""
        if motion is None:
            self._known.pop(key, None)
        else:
            self._known[key] = ((float(motion[0]) % 360.0, float(motion[1])), note)
        self._refresh()

    def _effective(self) -> Tuple[Optional[Tuple[float, float]], str]:
        key = self.source.currentData()
        if key == "manual":
            return (float(self.heading.value()), float(self.speed.value())), "typed in"
        order = [key] + [k for _l, k in SOURCES if k not in (key, "manual")]
        for k in order:
            if k in self._known:
                m, note = self._known[k]
                label = dict((kk, ll) for ll, kk in SOURCES).get(k, k)
                return m, (note or label) + ("" if k == key else f" ({label}; the chosen source has none)")
        return None, "no motion yet: pick a storm, load recon, or type one in"

    def motion(self) -> Optional[Tuple[float, float]]:
        return self._effective()[0]

    def _refresh(self, *_):
        m, note = self._effective()
        manual = self.source.currentData() == "manual"
        for w in (self.heading, self.speed):
            w.setEnabled(manual)
        if m is not None and not manual:
            for w, v in ((self.heading, int(round(m[0])) % 360), (self.speed, m[1])):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
        self.note.setText(f"{m[0]:03.0f}° at {m[1]:.0f} kt — {note}" if m else note)
        self.changed.emit()

    def _typed(self, *_):
        if self.source.currentData() == "manual":
            self._refresh()

    def state(self) -> dict:
        return {"source": self.source.currentData(), "heading": self.heading.value(), "kt": self.speed.value()}

    def restore(self, st: dict):
        if not st:
            return
        i = self.source.findData(st.get("source"))
        if i >= 0:
            self.source.setCurrentIndex(i)
        self.heading.setValue(int(st.get("heading", 0)))
        self.speed.setValue(float(st.get("kt", 0.0)))

#!/usr/bin/env python3
"""
recon10s_radarvolpanel — Radar › Radar volume.

A NEXRAD scan or JMA cycle gridded in 3-D around the storm
(recon10s_nexgrid), shown with the airborne-TDR viewer: CAPPIs at any
height, vertical sections through the eye at any bearing or along any line
dragged across the plan view, and azimuthal means about the eye. "Open in 3D
structure" hands the volume to Experimental › 3D structure for reflectivity
isosurfaces.
"""
from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
)

import recon10s_nexgrid as ng
from recon10s_tdrgridpanel import TdrGridPanel, _Job


class RadarVolumePanel(TdrGridPanel):
    send_to_3d = pyqtSignal(object)          # the gridded volume

    def __init__(self, parent=None, theme: str = "dark"):
        self.source_vol = None
        self.source_centre = None
        self.source_storm = ""
        self.source_motion = None
        self.raw_grid = None              # as gridded, before any re-centring
        super().__init__(parent, theme)
        for w in (self.chk_vectors, self.chk_kt, self.btn_in, self.btn_out):
            w.setVisible(False)                      # no 3-D winds from one radar; velocities are knots
        self.about.setText("NEXRAD Level II (unidata-nexrad-level2) or a JMA cycle, gridded here; the same "
                           "viewer as Radar › Airborne TDR.")
        self.chk_map.setToolTip("Draw this field at this height, georeferenced, over the\n"
                                "Imagery › Satellite loop.")
        self.status.setText("Build a volume from Radar › NEXRAD or JMA.")
        self._placeholder("No radar volume yet — load a scan under NEXRAD or JMA and press\n"
                          "“Build 3-D volume”")

    # ---------------------------------------------------------------- building

    def _build_source_box(self) -> QGroupBox:
        box = QGroupBox("Radar volume")
        f = QFormLayout(box)
        self.source_label = QLabel("Nothing gridded yet.")
        self.source_label.setObjectName("hint")
        self.source_label.setWordWrap(True)
        f.addRow(self.source_label)
        self.half_width = QSpinBox()
        self.half_width.setRange(40, 250)
        self.half_width.setSingleStep(10)
        self.half_width.setValue(150)
        self.half_width.setSuffix(" km")
        f.addRow("Half-width", self.half_width)
        self.resolution = QComboBox()
        for label, km in (("0.5 km", 0.5), ("1 km", 1.0), ("2 km", 2.0)):
            self.resolution.addItem(label, km)
        self.resolution.setCurrentIndex(1)
        f.addRow("Grid spacing", self.resolution)
        self.top = QDoubleSpinBox()
        self.top.setRange(6.0, 20.0)
        self.top.setSingleStep(1.0)
        self.top.setValue(15.0)
        self.top.setSuffix(" km")
        f.addRow("Up to", self.top)
        self.chk_eye = QCheckBox("Centre on the eye when there is one")
        self.chk_eye.setChecked(True)
        self.chk_eye.setToolTip("The clear eye within 40 km of the storm position, enclosed by echo at\n"
                                "1\u20133 km: sections and azimuthal means are then about the real centre,\n"
                                "not the advisory position. Where no eyewall encloses it, nothing moves.")
        f.addRow(self.chk_eye)
        row = QHBoxLayout()
        self.btn_rebuild = QPushButton("Rebuild")
        self.btn_rebuild.setEnabled(False)
        self.btn_rebuild.clicked.connect(self.rebuild)
        row.addWidget(self.btn_rebuild)
        self.btn_eye = QPushButton("Find the eye")
        self.btn_eye.setEnabled(False)
        self.btn_eye.clicked.connect(self.find_eye)
        row.addWidget(self.btn_eye)
        self.btn_pick = QPushButton("Click a centre")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setEnabled(False)
        self.btn_pick.setToolTip("Click the plan view where the centre is.")
        row.addWidget(self.btn_pick)
        f.addRow(row)
        self.btn_3d = QPushButton("Open in Experimental › 3D structure")
        self.btn_3d.setObjectName("primary")
        self.btn_3d.setEnabled(False)
        self.btn_3d.clicked.connect(lambda: self.send_to_3d.emit(self.grid) if self.grid is not None else None)
        f.addRow(self.btn_3d)
        note = QLabel("Every elevation of the scan, interpolated between beams to 0.5 km levels; nothing is "
                      "filled where neighbouring beams are more than 2 km apart. Radial velocity is the "
                      "component along the beam, so sections show it but there are no 3-D winds.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        f.addRow(note)
        return box

    # ------------------------------------------------------------------ data

    def build_from(self, vol, centre: Optional[tuple], storm: str = "", motion=None):
        """Grid a NEXRAD Volume or JmaVolume about `centre` (lat, lon; the radar if None)."""
        self.source_vol, self.source_centre = vol, centre
        self.source_storm, self.source_motion = storm or "", motion
        self.rebuild()

    def rebuild(self):
        vol = self.source_vol
        if vol is None:
            return
        for b in (self.btn_rebuild, self.btn_eye, self.btn_3d):
            b.setEnabled(False)
        site = getattr(vol, "site_label", None) or vol.scan.site
        self.source_label.setText(f"Gridding {site} {vol.scan.time:%d %b %Y %H:%M:%SZ}…")
        hw, dx, top = float(self.half_width.value()), float(self.resolution.currentData()), self.top.value()
        z = [0.5 * k for k in range(1, int(round(top / 0.5)) + 1)]
        eye = self.chk_eye.isChecked()
        dealias = not hasattr(vol, "refl")            # JMA velocities arrive unfolded

        def work(status, vol, centre, storm, motion):
            grid = ng.grid_volume(vol, centre, half_width_km=hw, dx_km=dx, z_km=z, motion=motion, dealias=dealias,
                                  storm=storm, progress=status)
            found = ng.find_eye(grid) if eye else None
            return grid, found
        job = _Job(work, vol, self.source_centre, self.source_storm, self.source_motion)
        job.sig.status.connect(self.status.setText)

        def done(res):
            grid, found = res
            self.raw_grid = grid
            note = ""
            if found is not None:
                grid = ng.recentre(grid, *found)
                note = f" Centred on the eye, {math.hypot(*found):.0f} km from the first guess."
            elif eye:
                note = " No eye found; centred on the storm position." if self.source_centre else ""
            self._after(grid, note)
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (self.status.setText(m), self.btn_rebuild.setEnabled(True)))
        self.pool.start(job)

    def _after(self, grid, note: str = ""):
        vol = self.source_vol
        site = getattr(vol, "site_label", None) or vol.scan.site
        fields = ", ".join(ng.field_label(k)[0].lower() for k in grid.fields)
        mot = (f" Storm-relative velocity with {self.source_motion[0]:03.0f}°/{self.source_motion[1]:.0f} kt "
               "removed." if self.source_motion and "SR_VELOCITY" in grid.fields else "")
        self.source_label.setText(f"{site}, {vol.scan.time:%d %b %Y %H:%M:%SZ}: {fields}. Centre "
                                  f"{grid.centre[0]:.2f}, {grid.centre[1]:.2f}.{note}{mot}")
        for b in (self.btn_rebuild, self.btn_eye, self.btn_3d, self.btn_pick):
            b.setEnabled(True)
        if self.grid is None:
            self.view_box.blockSignals(True)
            self.view_box.setCurrentIndex(self.view_box.findData("overview"))
            self.view_box.blockSignals(False)
        self.show_grid(grid)

    def find_eye(self):
        base = self.raw_grid
        if base is None:
            return
        # search around the current centre, in the as-gridded frame
        cx, cy = self._offset()
        moved = ng.recentre(base, cx, cy)
        found = ng.find_eye(moved)
        if found is None:
            self.status.setText("No eye found: nothing within 40 km of the current centre is enclosed by "
                                "echo at 1\u20133 km.")
            return
        self._after(ng.recentre(base, cx + found[0], cy + found[1]),
                    f" Eye found {math.hypot(*found):.0f} km from the previous centre.")

    def _offset(self):
        """The current centre's position in the as-gridded frame (km)."""
        if self.raw_grid is None or self.grid is None:
            return 0.0, 0.0
        return float(self.raw_grid.x[0] - self.grid.x[0]), float(self.raw_grid.y[0] - self.grid.y[0])

    def _on_click(self, event):
        if self.btn_pick.isChecked() and event.button == 1 and event.xdata is not None:
            role = self._roles.get(event.inaxes)
            if role is not None and role[0] == "plan" and self.raw_grid is not None:
                cx, cy = self._offset()
                self.btn_pick.setChecked(False)
                self._after(ng.recentre(self.raw_grid, cx + event.xdata, cy + event.ydata), " Centre picked by hand.")
                return
        super()._on_click(event)

    # nothing to look up: missions belong to the Airborne TDR tab
    def set_storm_name(self, name: str):
        pass

    def state(self) -> dict:
        return {"view": self.view_box.currentIndex(), "field": self.field_box.currentData(),
                "height": self.height.value(), "half_width": self.half_width.value(),
                "dx": self.resolution.currentIndex(), "top": self.top.value(), "eye": self.chk_eye.isChecked()}

    def restore(self, st: dict):
        if not st:
            return
        self.view_box.setCurrentIndex(int(st.get("view", 0)))
        self.height.setValue(int(st.get("height", 4)))
        self.half_width.setValue(int(st.get("half_width", 150)))
        self.resolution.setCurrentIndex(int(st.get("dx", 1)))
        self.top.setValue(float(st.get("top", 15.0)))
        self.chk_eye.setChecked(bool(st.get("eye", True)))

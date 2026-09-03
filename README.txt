RECON10s (IWG1 to HDOB Converter, Plotter & Live Monitor)
=========================================================
Version 1.2.1

Converts IWG1 aircraft data from the NOAA Hurricane Hunters into the HDOB
format used for NHC recon products, plots it on a Mercator map, and watches a
mission live on a stack of time series.

----------------------------
What's new in 1.2.1
----------------------------

- The tkinter GUI is gone. recon10s_qt.py is the interface now, and nothing in
  the package needs tkinter -- the map is pinned to matplotlib's Qt backend.
- Extrapolated MSLP is drawn the normal way up, so a deepening centre dips
  toward the bottom of the panel instead of climbing.
- Two more panels, stacked under the wind panel and sharing its time axis:
    * Temperature and dew point (degC). Opens at 0-25 and stretches only if
      the data leaves that range.
    * Geopotential height (m) and static pressure (mb, inverted so it tracks
      height). Ragged traces here are the turbulence signature.
  Every series has its own tick box; a panel with nothing shown collapses.
  One crosshair reads all three panels at once.
- Version is read from the code rather than the settings file. Older builds
  cached it in recon10s_settings.json, where it went stale and kept reporting
  1.1.1 no matter what was actually installed. That key is now ignored and
  removed on load.

----------------------------
What arrived in 1.2.0
----------------------------

- Live plot with auto-update on a timer you set (5-900 s, 30 s by default),
  at full 1 Hz resolution.
- Reading is roughly 12x quicker. Parsing had been running through a
  ThreadPool; it is pure Python and GIL-bound, so the pool cost more than the
  work (9.4 s with 4 workers vs 2.1 s with 1 on a 42 MB file). A single pass
  with a fast timestamp reader does the same file in about 0.7 s, with
  byte-identical HDOB output.
- Downloads are cached and resumed with HTTP Range. Mission files are appended
  to as the plane flies, so re-reading a flight in progress transfers only the
  minutes that are new. The live plot and the converter share one cached copy.
- A UTC window (--start/--end) filters on the timestamp before the rest of the
  line is parsed, which makes pulling 20 minutes out of a long mission about
  40x quicker.
- "Convert the data already loaded in the Live tab" skips the download and
  converts what is in memory.

----------------------------
Required Python Packages
----------------------------

- numpy
- matplotlib
- cartopy
- requests
- PyQt6
- pyqtgraph

----------------------------
Installation
----------------------------

1. Install Python 3.9+.
2. Install the packages:

   pip install numpy matplotlib cartopy requests PyQt6 pyqtgraph

   Or run install_deps.py if you would rather not open a terminal.

----------------------------
Usage
----------------------------

   python recon10s_qt.py

1. Paste an IWG1 URL, or browse to a local file, at the top of the window.
2. "Live plot" tracks the mission as it flies. Hover for a readout, click to
   pin it, click again to release.
3. "Convert to HDOB" writes the .txt and can open the map afterwards.

Command line:

   python recon10s.py --url <URL> --out hdob.txt --interval 30
   python recon10s.py --path flight.txt --out hdob.txt --start 11:00 --end 11:20

   --no-cache      always download the whole file
   --cache-dir     where to keep downloaded IWG1 files
   --workers       still accepted, now ignored (parsing is single-pass)

----------------------------
Notes
----------------------------

- Dark and light themes for both the live panels and the map.
- The extrapolated MSLP on the live plot is the same number that lands in the
  HDOB line, bias correction included, and is left blank above the 550 mb
  level where the converter reports a D-value instead.
- The outputs folder is just a convenient place for conversions; any folder
  works.
- Downloads are cached in your system temp folder. The Settings tab shows how
  much is there and can clear it.

----------------------------
Getting Recon Data
----------------------------

NOAA/AOML has IWG1 (Aircraft Data) files in their backend in this directory.
It is technically private though, as it's marked as Controlled Unclassified
Information (CUI), so use it carefully and at your own risk:

- <https://seb.omao.noaa.gov/pub/flight/aamps_ingest/iwg1/>

RECON10s (IWG1 to HDOB Converter, Plotter & Live Monitor)
=========================================================
Version 1.7.1

Converts IWG1 aircraft data from the NOAA Hurricane Hunters into the HDOB
format used for NHC recon products, plots it on a Mercator map, and watches a
mission live on a stack of time series.

----------------------------
What's new in 1.7.1
----------------------------

Satellite layout
  - Storm and satellite settings, the layer toggles and "Recon on the map"
    now sit in one column on the left (it scrolls on short screens), and the
    map, loop controls and status take the rest of the tab. The old strip of
    controls across the top left the map a short, wide band -- and a map is
    limited by its shorter side. On a 1920x1080 screen the map goes from
    about 350 to about 700 pixels across.
  - The map keeps fixed pixel margins for its title and labels instead of
    matplotlib's 10-12% defaults, re-worked out when the window is resized.
  - The colour bar is attached to the map's own edge, so the map stays
    centred instead of being pushed against the right of the figure.
  - With nothing to zoom to, the view is clipped to a mesoscale sector's
    coverage rather than framing empty space past its edge.

----------------------------
What arrived in 1.7.0
----------------------------

A home page and two sections
  - recon10s opens on a home page with three tiles: Reconnaissance (Live
    plot, Convert IWG1 to HDOB), Imagery (Satellite, Microwave) and
    Settings. Each section has a bar with Home and jump links; the Go menu
    has shortcuts (Ctrl+0 home, Ctrl+1 Reconnaissance, Ctrl+2 Imagery).
  - "Convert to HDOB" is now "Convert IWG1 to HDOB".

Missions that cross 00Z
  - HDOB messages are now dated the way NHC dates them, checked against
    their archive: each message's mission line carries the date of its
    FIRST ob, and its WMO header (URNT15 KNHC DDHHMM) is the time of its
    LAST ob. Before, every message repeated one date and one header time,
    so a flight past midnight was dated a day early from 00Z on. The
    "Storm date" field is gone -- the IWG1 timestamps already carry dates.
  - The UTC window takes a date on each end, with Yesterday / Today
    buttons, and "Whole flight" to clear it. A time-only window repeats
    every day, which quietly takes the same hours from both days of a long
    mission; a dated one does not. The line under it shows the resolved
    span, and an end before the start is refused.
  - Command line: --start / --end accept "YYYY-MM-DD HH:MM", or a time
    with the new --start-date / --end-date. A bare time still works the
    old way. --storm-date is accepted and ignored.

NHC HDOBs on the Satellite map
  - A "Recon on the map" sidebar in the Satellite tab. "Load NHC HDOBs"
    pulls every HDOB batch NHC published for the storm in the last N hours
    (AHONT1 Atlantic, AHOPN1 Pacific) -- USAF Reserve flights, which the
    NOAA IWG1 feed does not have, and NOAA flights too.
  - Batches are listed by mission (e.g. "AF302 0515E NOLO, HDOB 01-54,
    12:36-21:36Z"), each batch with its times and peak wind. Tick or untick
    missions or single batches; "Newest mission" ticks only the latest.
  - Plotted as wind barbs like the converted HDOB, which can be shown at
    the same time. Each flight has its own track line, broken where a batch
    is missing rather than drawn straight across the storm. Frame sync and
    the corner "Latest ob" label cover all of it.
  - Obs are decoded by position, so the extrapolated surface pressure reads
    correctly whether it is 989.7 mb ("9897") or 1004.2 mb ("0042").
  - Dropsondes moved into the same sidebar.

----------------------------
What arrived in 1.6.0
----------------------------

Dropsondes on the Satellite map
  - "Load dropsondes" (and Fetch imagery, while "Dropsondes" is ticked)
    pulls the storm's TEMP DROP messages from NHC's recon archive: REPNT3
    for the Atlantic, REPPN3 for the East/Central Pacific. With an HDOB
    loaded it covers the flight's span; otherwise the last 12 hours.
  - Each drop is a numbered white dot at its release point, 1..N by release
    time across everything loaded, with a short line to where it splashed.
    Numbers stay fixed as the loop plays; with "Obs up to frame time" on,
    drops appear as the flight reaches them.
  - Click a dot for its sounding window. Where several dots sit under the
    cursor -- repeat passes through the same eyewall, or the crowded eye --
    a small menu lists them to choose from.

The dropsonde window
  - Skew-T (temperature, dew point, isotherms, dry and moist adiabats) with
    wind barbs, the mandatory-level table (height, wind, temperature, RH),
    the significant-wind table, and a GOES inset from the scan just before
    the drop, with the release point, splash track and storm centre.
  - Mean wind in the lowest 500 m and 150 m straight from the message, with
    the usual reductions to the surface (x0.80 and x0.83).
  - Location is the message's own tag ("EYEWALL 315" -> "NW Eyewall",
    "CENTER", "EYE"); untagged drops get a bearing and distance from the
    centre, taken from the nearest centre drop within three hours when there
    is one, which beats an NHC fix from hours earlier.
  - Previous/next steps through the drops by number; "Raw message" shows
    the TEMP DROP text the window was decoded from. Several windows can be
    open at once. Save to PNG from the toolbar.

Decoding notes
  - Checked against a real Hurricane Nolo drop (AF302, 26 Sep 17:45:34Z):
    every table value, all 13 significant wind levels and both layer
    winds match NHC's message.
  - Levels above the header's wind indicator carry no wind group, the
    100+ kt speeds ride in the direction's last digit, and the 62626
    remarks are wrapped mid-token ("WL150 3" / "1001") -- all handled.

----------------------------
What arrived in 1.5.0
----------------------------

Satellite has its own tab
  - The HDOB tab is back to full-width text. Converting sends the obs to the
    Satellite tab ("Send to the Satellite tab", on by default) without
    switching away from the text; "Open an existing HDOB..." does switch.

Every ob is a wind barb
  - The separate dots and barbs are gone: each HDOB ob is drawn as one barb,
    coloured by its speed with the same buckets as before, over a thin dark
    outline so pale barbs still read against white cloud tops.
  - "Declutter barbs" drops barbs that would overlap on screen. It works in
    screen space, so zooming in with the toolbar brings them back, and the
    newest ob is always kept. Off by default -- every ob is drawn.

Loops of 15 or 30 frames
  - Pick "Latest only", "15 frames" or "30 frames" and Fetch. Frames load
    six at a time and appear as they arrive, so the slider works before the
    last one lands. A 30-frame full-disk loop (5 hours) takes ~10 s with the
    first frame up in ~2 s; a 30-frame meso loop (30 minutes) ~3 s.
  - Slider, previous/next, and play/pause. Playback pauses briefly on the
    newest frame before looping. Scrubbing only swaps the image and the barbs
    rather than redrawing the map.
  - Top-left corner of the map: the frame's scan time and the time of the
    latest barb drawn. With "Obs up to frame time" on, each frame shows only
    the obs taken by then, so the loop replays the flight. The newest frame
    always shows every ob, since the aircraft is usually ahead of the
    imagery -- the corner label shows that gap in minutes.
  - Mesoscale sectors can be moved during a loop. Frames from when the
    sector was pointed elsewhere are dropped and counted in the status line.

Fixes
  - Obs after 00Z were dated to the wrong day, because the converter repeats
    the storm date on every HDOB message. Dates now come from the running
    sequence of times, so a pass across midnight stays in order.
  - GOES-18's field of view straddles the dateline (141.7E round to 55.7W),
    and the coverage test treated that as covering nothing. A GOES-18 meso
    sector over a Central Pacific storm would never have been picked.

----------------------------
What arrived in 1.4.0
----------------------------

Microwave tab: real polar-orbiter passes over a storm
  - GMI (GPM), AMSR3 (GOSAT-GW) and MWI (WSF-M) from the PPS near-real-time
    server, jsimpsonhttps.pps.eosdis.nasa.gov. Needs an account registered
    for NRT access; on that server the email is both username and password.
  - Sign-in is kept for the session only unless you press "Save to
    jsimpson.json". That file is plain text beside the scripts, and it is
    listed in .gitignore so it cannot end up in a commit by accident. "Forget"
    deletes it.
  - Finding passes does not download everything in the look-back window.
    Each satellite's orbit is propagated (SGP4, TLEs from CelesTrak, cached
    for a day, with a bundled fallback) across every granule's time span,
    and only granules whose track reaches the storm are listed. For GMI's
    five-minute granules that is typically 2 of ~140 in twelve hours.
  - Two maps side by side, 89 GHz and 37 GHz, each with tabs for Color, H,
    V and PCT. All of it follows NRL's definitions from the GeoIPS source:
        color89  R = 1.818V-0.818H (220-310 K, inverted), G = H, B = V
        color37  R = 2.181V-1.181H (260-280 K, inverted), G = V, B = H
        89PCT    1.7V - 0.7H      37PCT  2.15V - 1.15H
    The colour composites and the standalone PCTs use different weights on
    purpose; NRL kept the older ones inside the colour recipes. H, V and PCT
    use NRL's 89H / 89PCT / 37H / 37PCT colour tables.
  - Pixel probe: hover over either map for the H, V and PCT of the nearest
    native footprint -- the value the instrument recorded, not a resampled
    one -- plus its scan and pixel number. Click to pin, click to release.
  - Channels are found from each file's own metadata (the Tc LongName in
    every swath group), not from hardcoded positions, since GMI, AMSR3 and
    MWI lay their channels out differently. A file that does not describe
    its channels is refused with a list of what it does contain, rather
    than shown with the wrong channel labelled 89 GHz.

----------------------------
What arrived in 1.3.0
----------------------------

Map beside the HDOB text
  - The "Convert to HDOB" tab is now split: HDOB text on the left, map on
    the right. No more separate map window. Drag the divider to resize.

GOES-18 / GOES-19 imagery under the recon values
  - Bands 2 (red visible), 7 (shortwave IR), 9 (mid-level water vapour) and
    13 (clean longwave IR), full disk or mesoscale, straight from the NOAA
    public S3 buckets. No AWS account needed.
  - The storm position comes from NHC's CurrentStorms feed, falling back to
    the ATCF working best track (b-deck), and finally to the middle of the
    recon track itself if NHC has nothing.
  - That position picks the satellite (whichever subpoint is nearer: GOES-19
    at 75.2W, GOES-18 at 137W) and the sector. Mesoscale sectors are
    movable and their location is only recorded inside the file, so "Auto"
    reads the header of the newest M1 and M2 and uses one if it covers the
    storm -- meso refreshes every minute against the full disk's ten.
    Otherwise it falls back to the full disk. Any choice can be forced.
  - Only the part of the image around the storm is downloaded, using HTTP
    Range requests. A full-disk band 2 file is ~230 MB; opening it costs
    about 1 MB and a storm-sized cut about 20-35 MB. Band 13 cuts are ~5 MB.
  - Layers you can toggle: imagery, track, values (points coloured by wind,
    same colours as the old map), wind barbs, colour bar, NHC fix marker.
    "Zoom to recon" frames the map on the flight track.

IR colour table
  - Bands 7, 9 and 13 use the enhancement curve in recon10s_irtable.py,
    sampled from the reference colour bar. Note it is not linear: 20 degC
    per tick from +40 to -20, 10 degC per tick from -20 to -90, and stepped
    in roughly 1 degC blocks below -19. Band 2 is shown as a square-root
    stretched grey.

----------------------------
What arrived in 1.2.1
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
- h5py           (reading GOES and microwave files)
- scipy          (microwave display and pixel probe)
- sgp4           (orbit screening for microwave passes)
- pyproj         (usually installed with cartopy)

----------------------------
Installation
----------------------------

1. Install Python 3.9+.
2. Install the packages:

   pip install numpy matplotlib cartopy requests PyQt6 pyqtgraph h5py scipy sgp4

   Or run install_deps.py if you would rather not open a terminal.

----------------------------
Usage
----------------------------

   python recon10s_qt.py

1. From the home page, open Reconnaissance. Paste an IWG1 URL, or browse
   to a local file, at the top.
2. "Live plot" tracks the mission as it flies. Hover for a readout, click to
   pin it, click again to release.
3. "Convert IWG1 to HDOB" writes the .txt and sends the obs to Imagery.
4. In Imagery > Satellite, pick a storm, band, sector and frame count, then
   "Fetch imagery". Load NHC HDOBs and dropsondes from the sidebar. Scrub
   with the slider or press play.

Command line:

   python recon10s.py --url <URL> --out hdob.txt --interval 30
   python recon10s.py --path flight.txt --out hdob.txt --start 11:00 --end 11:20
   python recon10s.py --path flight.txt --out hdob.txt \
          --start "2026-09-26 23:30" --end "2026-09-27 00:30"

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
- The first map draw downloads coastline data for cartopy (once), so it
  needs internet and pauses briefly.
- Downloads are cached in your system temp folder. The Settings tab shows how
  much is there and can clear it.

----------------------------
Microwave data access
----------------------------

Register for near-real-time access at
https://registration.pps.eosdis.nasa.gov/registration/ -- the NRT option is
separate from a normal Earthdata login. Granules are cached in your system
temp folder once downloaded, so reopening a pass is instant.

----------------------------
Getting Recon Data
----------------------------

NOAA/AOML has IWG1 (Aircraft Data) files in their backend in this directory.
It is technically private though, as it's marked as Controlled Unclassified
Information (CUI), so use it carefully and at your own risk:

- <https://seb.omao.noaa.gov/pub/flight/aamps_ingest/iwg1/>

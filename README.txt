RECON10s (IWG1 to HDOB Converter, Plotter & Live Monitor)
=========================================================
Version 2.6.0

Converts IWG1 aircraft data from the NOAA Hurricane Hunters into the HDOB
format used for NHC recon products, plots it on a Mercator map, and watches a
mission live on a stack of time series.

----------------------------
What's new in 2.6.0
----------------------------

Himawari-9 (Imagery > Satellite)
  - Himawari-9 imagery for storms west of the dateline: the West Pacific,
    the South Pacific and Australian region, and the eastern Indian Ocean
    (the Bay of Bengal; the Arabian Sea and the south-west Indian Ocean are
    beyond its useful view, and Meteosat, which sees them, is not on NOAA's
    open buckets). "Auto by longitude" now chooses among GOES-19, GOES-18
    and Himawari-9 -- Himawari west of about 178W -- or pick "Himawari-9
    (west Pacific)" yourself.
  - From NOAA's open copy of JMA's data on AWS (noaa-himawari9, the AWIPS
    tiles), no sign-in. The full disk every 10 minutes comes in 88 tiles,
    of which a storm needs one to four (about 0.3 MB each for the IR
    bands): a 15-frame loop is a few MB. The target area -- the 1000-km box
    JMA scans every 2.5 minutes wherever it is watching, usually the
    strongest typhoon -- is used like a GOES mesoscale sector: "Auto" takes
    it whenever it is over the storm, or choose "Target area (Himawari)".
    Champi's 15-frame loop on 18 Oct came from it in 5 seconds and 5 MB.
  - The same imagery list as for GOES, band for band by wavelength: red
    visible (0.5 km), shortwave IR, the three water-vapour bands, the IR
    window bands, and the WV - IR, Airmass and Night Microphysics
    composites. Himawari's band 13 is 10.4 um (ABI's 10.3).
  - The archive runs from December 2022 (when Himawari-9 took over), so
    saved cases and "Imagery at this time" replay West Pacific storms too.
    NOAA's copy has gaps (12 Oct - 25 Nov 2025, for one).
  - GLM lightning stays GOES's: it reaches the Pacific to about 170E, and
    Himawari has no lightning mapper. A storm beyond every satellite's view
    says so instead of drawing an empty map; one seen obliquely (zenith
    angle over about 70 degrees) is flagged in the map's title.

Every basin in the storm lists
  - The storm lists (Satellite and Microwave, and through them every tab
    that follows the chosen storm) now hold every active system worldwide:
    NHC's, then JTWC's -- West Pacific, North Indian Ocean, Southern
    Hemisphere -- then invests in any basin: "Typhoon Champi (29W)",
    "Tropical Storm Wati (01P)", "Invest 96A".
  - JTWC's storms come from knackwx's open ATCF feed (api.knackwx.com/atcf/v2:
    position, intensity, pressure and motion for every system, refreshed
    each cycle), or NRL's GeoIPS sector file when knackwx cannot be
    reached. NHC's own list stays the authority for its basins; if NHC is
    down, knackwx supplies those storms too. The JMA radar tab takes its
    typhoons from knackwx first as well (the sector file runs a cycle
    behind).
  - Best tracks for JTWC's storms: knackwx's working b-deck for an active
    storm (the newest cycle), else UCAR's mirror of JTWC's b-decks, else
    natyphoon.top's copy -- each checked to be the same storm and season,
    since knackwx's short ids (29W) carry no year. natyphoon could not be
    reached from where this release was built, so that last fallback is
    untested live.
  - CIMSS ADT, D-PRINT and D-MINT for West Pacific, Southern Hemisphere
    and Indian Ocean storms: the right letter (01P, not 01S) from the storm
    list; southern seasons filed under the year they end; a real-time ADT
    list left over from an earlier storm of the same number recognised by
    its dates and skipped.
  - Wherever a best track or advisory is shown (Satellite intensity,
    Intensity, Wind field) it is named for whose it is, NHC or JTWC.
    Forecast cones and model tracks remain NHC's only.

Satellite intensity (Experimental) in every basin
  - Estimates from Himawari-9 band 13 (GOES-18 east of about 178W) work
    exactly as from GOES. The network learnt only northern storms, so a
    southern storm is mirrored north-south (poleward stays up and the
    spiral turns the northern way) and given the size of its latitude: to
    the network it is then a northern storm in every respect it can see.
  - Mapped onto JTWC's best track. The network learnt NHC's best tracks,
    whose pressures come mostly from aircraft; JTWC's come from its
    wind-pressure relationship, in basins of lower environmental pressure,
    and came out about 5 mb lower than the network's for the same storms.
    So in JTWC's basins the pressure is moved by an amount that depends on
    the pressure itself and the latitude, and the ranges of wind and
    pressure are widened to hold the best track as often as they say. The
    wind is left as the network gives it: a shift fitted the same way
    helped weak systems but made tropical storms and typhoons worse. The
    readout says "mapped onto JTWC's best track". The basin input the
    network takes (Atlantic or eastern Pacific) is the eastern Pacific's,
    which scored far better (7.2 kt against 9.7 on 2020-2023).
  - No estimate from IR seen more obliquely than a 70-degree zenith angle
    (the south-west Indian Ocean and the Arabian Sea); the panel says why.

How it does there (storms it never saw)
  - TC PRIMED's storms of JTWC's basins, 2020-2025: 31,753 cases of 356
    storms, one per overpass image, with the archive's IR (Himawari's for
    the West Pacific and Australia) and JTWC's best track. The basin input
    and the pressure mapping were fixed on 2020-2023; these are 2024-2025,
    which nothing was fitted to: 9,307 estimates of 108 storms the app can
    make (zenith angle under 70 degrees).
  - Vmax mean error 6.3 kt (RMSE 8.2), 83 % within 10 kt; from tropical-
    storm strength up (6,034 estimates) 6.4 kt, RMSE 8.6. MSLP 4.7 mb
    (bias +0.8; 6.5 mb and +5.3 before the mapping). From IR alone 6.7 kt
    and 4.9 mb. By region: West Pacific 6.2 kt and 5.2 mb (4,675
    estimates), Australia and the South Pacific 6.2 kt and 4.0 mb (3,959),
    North Indian Ocean 5.4 kt and 4.0 mb (371); the eastern edge of the
    south-west Indian Ocean that is in reach, 8.3 kt and 5.8 mb (about
    300).
  - Against CIMSS on the same cases (each CIMSS estimate matched to the
    nearest of these within 45 minutes; 95 % intervals from resampling
    whole storms) -- CIMSS's run operationally in these basins:
      D-PRINT, 11,452 estimates of 101 storms: D-PRINT 7.4 kt and 5.3 mb;
        this model 6.3 kt (better by 1.1 kt, 0.5-1.7) and 5.0 mb with its
        newest pass, 6.8 kt (better by 0.6, 0.0-1.2) and 5.3 mb from IR
        alone.
      D-MINT, 3,176 estimates of 103 storms: D-MINT 7.1 kt and 5.2 mb; this
        model 6.2 kt (better by 0.9, 0.2-1.5) and 5.0 mb, and 6.7 kt from
        IR alone (no clear difference).
  - By strength, as for the Atlantic it comes out low in strong storms:
    64-95 kt 8.8 kt (bias -4.3), 96-136 kt 7.5 (bias -4.6). Below 34 kt it
    runs high (bias +5.9): JTWC's depressions are weaker than the network,
    raised on NHC's, makes them.
  - The ranges, widened as above: the 10-90 % range held JTWC's best track
    85 % of the time for wind and 81 % for pressure, the 25-75 % range 51 %
    and 52 % (63 % and 49 % for the 10-90 % before).
  - The app's Himawari-9 pipeline gives the IR TC PRIMED archives (which
    is Himawari's band 13 too): on 277 West Pacific cases of 2023-2025 it
    found the same scans to the minute, 0.1 K from the archive's images
    (1.5 K rms, median), the warmest eye pixel within 0.4 K and the images
    within about a kilometre of each other. The estimates from the app's IR
    scored 6.5 kt against 6.8 from the archive's on the same cases.
  - Seen obliquely: on 296 cases of 2023-2025 that Himawari sees at zenith
    angles of 55-86 degrees (mostly the south-west Indian Ocean and the
    Arabian Sea, where TC PRIMED's images are Meteosat's, seen squarely),
    estimates from the app's IR were as good as from the archive's up to 62
    degrees (8.3 kt against 9.0), a little worse from 62 to 70 (9.2
    against 7.9), and fell apart beyond -- 14 to 21 kt, 10-13 kt too low.
    Hence the 70-degree limit, and a note on estimates past 62.
  - Live, IR only (no PPS sign-in where this was built), through the app's
    own code, 18-19 Oct: Champi re-intensifying on Himawari-9 -- 93-105 kt
    from 15:30 to 18:30Z, when JTWC had 130 kt (ADT 112-115, D-PRINT
    92-108), then 112-131 kt and 924-931 mb to 02:30Z against JTWC's 135 kt
    and 926 mb at 00Z. Wati (01P, east of the dateline, so GOES-18,
    mirrored as a southern storm): 55-65 kt and 985-991 mb against JTWC's
    55-60 kt and 988-992 mb.

Tests
  - Himawari-9, offline: synthetic AWIPS tiles laid out as in NOAA's bucket
    behind a fake S3 -- listing, the tiles a storm needs, stitching across
    four of them with every pixel where the scene puts it, space left
    blank, a scan still arriving not offered, the target area found,
    chosen over the storm and trimmed, a composite from matching scans, the
    map title -- plus the tile numbering against real tiles' offsets, the
    satellite choice and the edge of the view, and the satellite-intensity
    grid from a Himawari image.
  - Storms in every basin: knackwx's feed (labels, short ids, a southern
    "typhoon" shown as a tropical cyclone), the merge with NHC's list and
    the fallbacks (NHC down; knackwx down, so the sector file), best tracks
    from knackwx, UCAR and natyphoon in that order with another season's
    29W refused, CIMSS codes and a stale ADT list skipped, and the JMA tab's
    typhoons.
  - Satellite intensity: a mirrored southern storm giving exactly the
    northern estimate, no estimate from an edge-on view, the mapping onto
    JTWC's best track.
  - GUI: a West Pacific storm in the lists, Himawari-9 and the target area
    offered, the other tabs following it with JTWC's numbers, no GLM that
    far west. 160 tests, all offline.

Data use
  - Himawari-9: the Japan Meteorological Agency's data, as re-served by
    NOAA's Open Data Dissemination program (s3://noaa-himawari9).
  - Storm lists and best tracks: knackwx.com's ATCF API, NRL's GeoIPS
    sector file, UCAR's open mirror of JTWC's b-decks and natyphoon.top;
    the best tracks are JTWC's (and NHC's).
  - The validation used TC PRIMED (cited under 2.5.0) and CIMSS's D-PRINT,
    D-MINT and ADT histories, fetched as published.

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
- scipy          (microwave display, TDR and radar grids)
- sgp4           (orbit screening for microwave passes)
- pyproj         (usually installed with cartopy)
- pillow         (loop export)
- arm_pyart      (NEXRAD Level II)
- scikit-image   (3D isosurfaces)

----------------------------
Installation
----------------------------

1. Install Python 3.9+.
2. Install the packages:

   pip install numpy matplotlib cartopy requests PyQt6 pyqtgraph h5py scipy sgp4 \
               pillow arm_pyart scikit-image

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
5. The loaded recon feeds Reconnaissance > Analysis, Wind field and
   Intensity, and Experimental > Synthetic TDR. File > Save case keeps it
   all; File > Open case replays it.

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

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
What arrived in 2.5.0
----------------------------

Satellite intensity (Experimental > Satellite intensity)
  - recon10s's own satellite estimates of a storm's maximum wind and
    central pressure, from a convolutional network trained here: nothing to
    train on your PC and nothing to import. Like CIMSS's D-PRINT (IR only)
    and D-MINT (IR plus microwave) it looks at the storm itself rather than
    working through a Dvorak tree. It differs in two ways. D-MINT makes its
    estimates at the time of each pass; this one estimates at any time from
    the most recent pass of the last 12 hours, however old, together with
    the pass's age and the IR from the time of the pass, so the network can
    weigh the pass's structure against how the cloud tops have changed
    since. And one network covers both: with no pass (or no PPS sign-in)
    it works from IR alone, as it was trained to.
  - What it looks at: GOES-East or -West band 13 (whichever views the
    storm more squarely) on a 4-km grid 512 km across, centred on the
    storm, at the estimate time and 3, 6, 9 and 12 hours before; the newest
    GMI, AMSR3 or WSF-M pass (37 and 89 GHz, through the Microwave tab's
    PPS sign-in) on an 8-km grid; and latitude, motion, basin and local
    solar time. The centre comes from the recon fixes when there are any
    within 90 minutes, else NHC's working best track and advisory, carried
    along the motion for up to 6 hours past the last position.
  - What it gives: the median and the 25-75 % and 10-90 % ranges of the
    best-track maximum wind and central pressure, and beside them what the
    IR alone would have given, so you can see what the pass changed. Run it
    for the latest scan or any past time, and back through the hours
    before it (every 30 minutes, hourly or every 3 hours); "Add an estimate
    every 30 minutes" keeps it going. The chart draws the estimates and
    their ranges against NHC's best track and advisory and marks each pass
    used; below it are the IR and the 89-GHz PCT the newest estimate saw.
    "Copy all estimates (CSV)" puts the lot on the clipboard.
  - The estimates go to Reconnaissance > Intensity as "recon10s
    satellite": on the timeline, and in Verification against the recon or
    the best track. Saved cases keep them and replay them.

CIMSS D-PRINT and D-MINT (Reconnaissance > Intensity, and on the new chart)
  - Fetched with the best track and ADT for the storm, from CIMSS's
    history files: D-PRINT every hour or three from IR, D-MINT at each
    microwave pass. On the timeline, in Verification beside ADT and
    recon10s's estimates, and drawn on the Satellite intensity chart (a
    tick box hides them). Saved cases keep the files as published.

Removed
  - Opening an estimator's output file (TCFuse's CSV/JSON or any other) in
    Intensity > Verification: recon10s makes its own estimates now. Cases
    saved by 2.3.0-2.4.0 still open; an estimator file inside one is
    ignored.

How it was trained
  - TC PRIMED v01r01 (NOAA NESDIS / CIRA): every overpass of Atlantic,
    east and central Pacific storms 2000-2025 -- 98,185 files of 977
    storms -- each with the geostationary IR at its time (CIRA/RAMMB's
    tropical cyclone IR archive, 4 km) and the best track interpolated to
    it. Read over HTTP range requests (only the IR, the 37/89-GHz channels
    and the storm metadata of each file, about 127 GB) and reduced to the
    grids above.
  - Trained on 56,770 cases of 890 storms, 2000-2023, one per IR image;
    2024 and 2025 (4,632 cases, 66 storms) were held out of training
    entirely and used only to measure it, with the recipe fixed beforehand
    and the last epoch kept. The network: an IR branch and a microwave
    branch of convolutions and a small head that also takes the scalars,
    trained for the five quantiles (pinball loss). Each training case got
    its newest pass 40 % of the time and otherwise the newest pass of a
    random older age bin (0-2, 2-4, 4-7 or 7-12 h), and no pass at all 30 %
    of the time, so that old passes and no passes are as familiar to it as
    fresh ones; every case was turned to a random angle and shifted a few
    km. Each estimate in the app averages four 90-degree turns.
  - It ships as scripts/models/satint_v1.npz (3.3 MB) and runs in numpy
    (a fraction of a second per estimate; the GOES and PPS downloads take
    the time). training/satint/ holds the whole pipeline.
  - The app's own GOES pipeline gives the IR the network learnt from: on
    160 cases of 2025 it found the same scans to the minute, with a mean
    difference of 0.0 K from TC PRIMED's images, about 1 K rms (median),
    no offset in position, and the warmest eye pixel within a few tenths of
    a kelvin. On those cases (mostly hurricanes) the estimates from the
    app's IR scored 6.4 kt against 7.0 from TC PRIMED's own images.

How good it is (on years the model never saw)
  - 2024-2025, held out of training: 4,632 estimates of 66 storms, one
    per overpass image, against the best track TC PRIMED carries. Vmax
    mean error 5.4 kt (RMSE 7.3, bias -1.3), 85 % within 10 kt; MSLP 3.8
    mb (RMSE 5.4). From the IR alone: 5.9 kt and 4.1 mb. With only GMI
    passes (the one of the app's three sensors TC PRIMED has): 5.6 kt and
    3.9 mb.
  - Against CIMSS on the same cases, each CIMSS estimate matched to the
    nearest of these within 45 minutes (differences with 95 % intervals
    from resampling whole storms):
      D-PRINT, 7,332 estimates: D-PRINT 6.9 kt; this model 5.5 kt with its
        newest pass (better by 1.4 kt, 0.9-1.9) and 6.0 from IR alone
        (better by 0.9, 0.5-1.3). MSLP: 5.0 mb against 3.9.
      D-MINT, 2,427 estimates: D-MINT 6.3 kt; this model 5.6 kt with its
        newest pass (better by 0.6 kt, 0.2-1.1) and 6.2 from IR alone (the
        same). MSLP: 4.4 mb against 3.9.
    The gap is widest in the strongest storms: at 137 kt and up, 7.7 kt
    against D-PRINT's 10.4, and 9.3 against D-MINT's 14.0.
  - By strength: 34-63 kt 5.0 kt, 64-95 kt 6.7 (bias -4.1), 96-136 kt 6.7
    (bias -2.7), 137 kt and up 8.0 (bias -7.3). Like every satellite
    method it comes out low in the strongest storms, though less so than
    D-PRINT and D-MINT (at 137 kt and up on the same cases: -6.7 kt
    against D-PRINT's -9.3; -8.3 against D-MINT's -13.4).
  - What a pass is worth, by its age: given its newest pass from 0-2, 2-4,
    4-7 or 7-12 h before, 5.6, 5.3, 5.4 and 5.4 kt, against 6.1, 5.9, 5.9
    and 5.9 from IR alone on the same cases. A pass 7-12 hours old helps
    about as much as a fresh one: the IR since the pass carries it forward.
  - Rapid change is hardest: after a rise of 20 kt or more in the previous
    12 h, 8.1 kt (bias -4.4); after a fall of 10 kt or more, 7.8 (bias
    -3.5); steady, 4.9.
  - The ranges mean what they say: the 25-75 % range held the best track
    50 % of the time, the 10-90 % range 78 %.
  - 2026, through the app's own GOES pipeline, IR only (PPS microwave needs
    your sign-in), against NHC's working best track: 1,511 estimates every
    3 hours of 32 storms, 5.4 kt (RMSE 7.5, bias -2.4); MSLP 4.4 mb. At
    D-PRINT's times 5.5 kt against D-PRINT's 6.0 (better by 0.6 kt,
    0.1-1.1); at D-MINT's times 5.9 against D-MINT's 6.0, D-MINT having
    microwave; at ADT's times 6.6 against ADT's 11.3. At 137 kt and up (24
    estimates), 5.7 kt against D-PRINT's 11.0.
  - Trudy, 18 Oct, live and IR only: 77-90 kt through the 12 hours to
    05:30Z, ending at 81 kt and 975 mb; NHC had 85 kt and 976 mb at 06Z.

Data use
  - TC PRIMED: Razin, M. N., C. J. Slocum, J. A. Knaff, P. J. Brown, and
    M. M. Bell, 2023: Tropical Cyclone Precipitation, Infrared, Microwave,
    and Environmental Dataset (TC PRIMED). Bull. Amer. Meteor. Soc., 104,
    E1980-E1998, https://doi.org/10.1175/BAMS-D-21-0052.1; the dataset,
    https://doi.org/10.25921/dmy1-0595. D-PRINT, D-MINT and ADT are CIMSS
    (University of Wisconsin-Madison) products, fetched as published.

Elsewhere
  - Choosing a storm in Imagery > Satellite now reaches the other tabs at
    once (Intensity, Wind field, Satellite intensity, ...), recon or not.
    Before, they waited for recon to be loaded. A storm ID typed by hand in
    Intensity is left as it is.

Tests
  - Satellite intensity: the numpy network against a plain convolution,
    the four-turn averaging, IR gridding (centred and round, from a lat/lon
    grid and from a GOES fixed-grid image made in the test), microwave
    gridding and coverage at a swath edge, the centre from best track,
    advisory and recon fixes, a pass older than 12 h left out, only the
    passes the estimates use downloaded, the inputs exactly as training
    built them, and the shipped model loading and ranking an eye above
    clear air. D-PRINT/D-MINT parsing and fetching, and cases keeping them.
    GUI runs: Estimate for Isaias with GOES and PPS replaced by synthetic
    grids (chart, readout, CSV, CIMSS overlay, hand-off to Verification),
    and choosing a storm without recon reaching the other tabs. 134 tests,
    all offline.

----------------------------
What arrived in 2.4.0
----------------------------

Synthetic TDR (Experimental > Synthetic TDR)
  - The wind field of the storm as a P-3's tail Doppler radar would show
    it, built from the recon already loaded -- flight-level obs and
    dropsondes from any aircraft, USAF or NOAA -- by a model of TDR
    structure learnt from HRD's analyses. It is an estimate, not a radar
    observation, and every view and the 3-D tab say so.
  - Surface (10 m) view: the estimated surface wind with 34/50/64-kt
    contours, "Max 82 kt - 27 km NE of centre", the RMW, the recon track
    coloured by its winds, dropsondes, the storm motion and shear arrows.
    Anything farther than 80 km (adjustable) from every ob is blacked out:
    there the field is the model's guess from the storm's intensity, size,
    shear and motion. The Airborne TDR views come with it -- plan view at
    any height 0.5-12 km, sections, azimuthal mean, probe -- plus the
    uncertainty as a field of its own. "Show on the satellite map" and
    "Open in 3D structure" work as for real TDR.
  - Pick the time (any centre fix, or the newest ob near the storm) and
    the window (+-2 h by default). Obs are placed relative to the moving
    centre as in Wind field; older ones are moved with the storm, scaled
    by the trend of the passes' peak winds, and trusted less. SHIPS shear
    is fetched for the time, or typed in, or left out.
  - The intensity estimate is a regression on the synthetic field's peak
    1-km wind and the strongest flight-level wind the recon measured, fitted
    to NHC's best track; the surface map is the 1-km wind scaled so its
    peak is that estimate. When the obs do not come within 50 km and 1.5
    RMW of the centre, or see the RMW from fewer than two sides, the
    estimate is marked low confidence.

How it was trained (here -- nothing to train on your PC)
  - The model ships as scripts/models/syntdr_v1.npz (2.7 MB). It learnt
    from 1,811 TDR analyses: all 1,623 usable ones in TC-RADAR v3m
    (NOAA AOML/HRD, 1997-2024, single passes and mission composites, read
    from AOML's directory) and 188 of HRD's real-time analyses of 2025-2026
    (seb.omao.noaa.gov/pub/flight/radar; the G-IV's and partial ones left
    out). Each is reduced to the storm-relative tangential and radial wind
    over Vm, on radius / RMW, azimuth from the motion and height. The mean
    structure is a regression on intensity, size, shear along and across
    the motion, translation speed and latitude; the departures from it are
    60 patterns (EOFs) and the variance they leave.
  - Fitting it to recon finds the size, strength and patterns that best fit
    the obs, with an uncertainty at every point; the size is chosen by how
    well each candidate explains the obs, starting from the radius of the
    observed peak.
  - The NOAA P-3s' own flight-level data (IWG1) for the TDR passes,
    2018-2026, set how far to trust the obs against the prior, the
    intensity calibration and the uncertainty -- each measured on storms
    the model had not seen. IWG1 is marked Controlled Unclassified
    Information; nothing from it is in the model file except those fitted
    numbers. training/ holds the whole pipeline if you want to rebuild it.

How good it is (on storms the model never saw)
  - Intensity against NHC's best track, 1,158 TDR analyses of 70 storms
    (2018-2026, 34 kt and up, core sampled), each estimated from the P-3's
    flight-level data alone with the model and calibration trained without
    that storm: mean error 5.7 kt, RMSE 7.7 kt, median 4.6 kt, no bias; 84%
    within 10 kt and 53% within 5 kt. On 2025-2026 alone (172 passes, 13
    storms, model trained on 1997-2024 only): mean error 6.2 kt, 86% within
    10 kt, bias -1.8 kt. The flight-level peak x 0.90 on the same passes:
    mean error 9.7 kt, bias -8.8, 59% within 10 kt.
  - By intensity: 34-63 kt 5.2 kt (86% within 10), 64-95 kt 5.3 (89%),
    96-136 kt 6.7 (79%, bias -2.2), 137 kt and up 8.2 (64%, bias -4.4): the
    strongest storms come out a little low. With the core unsampled (59
    analyses): mean error 8.9 kt, RMSE 15.9.
  - The field against the TDR analysis of the same pass (0.5-12 km, within
    150 km): tangential wind RMSE 8.8 kt (6.3 kt at flight level, 7.9 kt
    below 1 km); 2025-2026: 9.7 kt. Median errors: Vm 3.0 kt, RMW 12%.
    Errors are largest in the eye and eyewall (12 kt rms inside half the
    RMW, 6-7 kt beyond twice it); the uncertainty field follows that.
  - Trudy, 15 Oct (USAF only; no TDR of Trudy, so nothing of it is in the
    training): 79 kt at 16:45Z, 82 kt at 18:15Z and at 18:48Z (the newest
    ob near the storm); NHC's best track 80 kt at 18Z. Peak 25-27 km NE-ENE
    of the centre, RMW 27 km. Each pass left out in turn: the winds it
    measured predicted to 6.6 kt.

Learning from live recon
  - "Check: leave each pass out" builds the analysis again without each
    pass (an inbound and an outbound leg) and scores it on the pass it did
    not see.
  - With "Rebuild as new recon arrives" (on by default after the first
    build), each new batch builds a new analysis, and the one before is
    scored on the obs it had not seen.
  - A real TDR analysis of the same storm opened in Radar > Airborne TDR
    within 3 h is compared with the synthetic one at 2 km.
  - Every estimate is matched with NHC's best track once the track reaches
    its time (interpolated between 6-hourly points, never extrapolated).
    The mean error of the matched pairs is added to future estimates,
    shrunk toward zero so that 20 pairs weigh as much as the training. All
    of it is kept in syntdr_live.json in the settings folder.

Data use
  - HRD asks users of TC-RADAR to tell its radar team lead, Dr. Paul Reasor
    (paul.reasor@noaa.gov), the nature of the project, and publications to
    cite Fischer, Rogers, Reasor and Gamache (2022), Mon. Wea. Rev., 150,
    2255-2278, https://doi.org/10.1175/MWR-D-21-0223.1.

Elsewhere
  - 3D structure takes grids without reflectivity (synthetic ones): wind-
    speed isosurfaces, and the summary says the column is the model's.

Tests
  - Synthetic TDR on a tiny hand-made model: recovering a vortex from its
    own figure-four, a full build from a synthetic flight (size, strength,
    vorticity, the calibrated intensity and the surface peak agreeing), the
    pass splitter, core coverage, the live calibration's matching and
    shrinkage, the comparison with real TDR, and the shipped model loading
    and recovering a vortex. A GUI run builds Isaias's synthetic TDR from
    the fixture recon with the network refused, through every view, the
    satellite layer and the 3-D tab. 128 tests, all offline.

----------------------------
What arrived in 2.3.0
----------------------------

Eight additions, all in the app, none using SFMR. Opening a saved case
(File > Open case, Ctrl+O) replays everything below as it arrived.

Wind field (Reconnaissance > Wind field)
  - Every ob placed relative to the moving centre. The centre track is a
    local linear fit through the vortex-message and derived fixes (1 h
    Gaussian weights), carried on along the storm motion for up to 3 h
    past the last fix, so a few km of scatter between aircraft does not
    zigzag the obs.
  - A 2-D wind analysis on a polar grid, flight level or surface estimate:
    Barnes weights with separate radial and around-the-circle scales, since
    the obs come along radial legs; blank where no leg reaches. North up or
    storm motion up; earth- or storm-relative barbs. The motion comes from
    the recon fixes, the NHC advisory, the best track, or typed in.
  - 34/50/64-kt wind radii by quadrant: the farthest surface estimate
    reaching each speed along the track and in the analysis ("+": still
    that strong where sampling ended), beside NHC's best-track radii for
    the nearest synoptic time. By default the 700 mb ratio eases from 0.90
    inside 1.5 x RMW to 0.80 beyond 3 x RMW (NHC's note: 0.91 in the
    eyewall, 0.78 in the outer vortex), since radii live in the outer
    vortex.
  - Isaias, recon within 3 h of 8 Oct 18Z: 34-kt radius NE 106 nm against
    NHC's 110 nm. The aircraft samples only part of each quadrant, so the
    other quadrants are what the legs happened to cover.

Surface winds without SFMR (Reconnaissance > Analysis > Surface winds)
  - Every dropsonde beside the flight-level wind at its release: flight
    level x the standard ratio (700 mb 0.90, 850 mb 0.80, 1000 ft 0.85),
    WL150 x 0.83, MBL x 0.80, the sonde's own surface wind, and the
    observed surface/flight-level ratio. Double-click a row for the
    sounding. The radial profiles draw each leg's surface estimate dashed.
  - A ratio belongs to a level: none is applied above 625 mb, or while the
    aircraft climbs or descends (pressure changing more than 10 mb a
    minute). A P-3 descending from 595 to 685 mb 150-160 nm north-east of
    Isaias had otherwise stretched the 34-kt radius to 159 nm.
  - Isaias, 7-8 Oct: for flight-level winds of 50 kt or more, the sondes'
    surface wind over the flight-level wind had a median of 0.88 (19
    drops), close to the standard 0.90.

Intensity (Reconnaissance > Intensity)
  - Pressure and wind over time from every source on one chart: vortex-
    message minimum pressures; centre-drop splash pressures, raw and 1 mb
    lower per 10 kt of surface wind (NHC's rule of thumb for a sonde that
    missed the very centre); each pass's peak flight-level wind x ratio;
    eyewall drops' WL150 x 0.83; NHC's working best track (b-deck) and the
    current advisory; and satellite estimates. Any series can be hidden;
    the recon period or the whole storm.

Satellite estimates against the recon (Intensity > Verification)
  - CIMSS ADT's history list is fetched for the storm. Your own estimator's
    output could be opened too (CSV/TSV, TCFuse's history CSV or
    <ATCF>_latest.json) -- removed in 2.5.0, which has its own estimates.
  - Truth is the recon -- each vortex message's pressure, and each pass's
    flight-level wind x ratio or eyewall WL150 x 0.83, whichever is larger
    -- or NHC's best track. Estimates are interpolated to each truth time
    (or the nearest within 60 minutes): bias, MAE and RMSE, a chart, and
    the pairs in a table.
  - Isaias, ADT against 20 recon fixes on 7-8 Oct: MSLP bias -2.5 mb (MAE
    4.1); Vmax bias +9.3 kt (MAE 9.7). The aircraft rarely samples the very
    strongest wind, so estimates somewhat above the recon value are
    expected. Against the best track (12 synoptic times): MSLP -1.4 mb,
    Vmax -5.6 kt.
  - ADT's fixes over land (scene LAND, every value 0.0) carry no estimate
    and are left out.

TDR evolution (Experimental > TDR evolution)
  - Every gridded TDR pass of one or more P-3 missions measured the same
    way: peak wind at 2 km and in the lowest 5 km, the azimuthal-mean
    tangential wind maximum, the RMW at 2 and 8 km, vortex tilt, inner-core
    shear (vortex removed, with its jackknife uncertainty and grade), and
    SHIPS for the nearest cycle. Plotted pass by pass, with arrows for the
    tilt and shear headings; double-click a pass to open it under Radar >
    Airborne TDR. Each pass is measured once and remembered on disk.
  - Isaias, 11 passes from 7 Oct 16:47Z to 8 Oct 19:41Z: 2 km peak wind
    48 -> 92 kt, tilt 16 -> 35 km, inner-core shear 10-33 kt toward 049-105
    deg. SHIPS had 14-23 kt toward 059-070 deg: the same direction, though
    the inner core's shear was larger on 8 Oct, 16-18Z (29-33 kt against
    14-17).

Radar volume: NEXRAD and JMA in 3-D (Radar > Radar volume)
  - "Build 3-D volume" (NEXRAD or JMA tab) grids every elevation of a scan
    around the storm -- 1 km apart (or 0.5/2), 0.5-15 km up -- and opens it
    in the Airborne TDR viewer: CAPPIs at any height, vertical sections
    through the centre at any bearing or along any line dragged across the
    plan view, and azimuthal means. "Open in Experimental > 3D structure"
    draws reflectivity isosurfaces (30/40/50 dBZ, lightly smoothed).
  - Gridding: the nearest ray and gate of each elevation, the beam's height
    from the 4/3-earth model, then linear interpolation between the beams
    above and below each level. Within half a beamwidth of a beam its own
    value is kept; nothing is filled where neighbouring beams are more than
    2 km apart, nor beyond the lowest and highest beams.
  - Centred on the storm at the scan time (recon fixes, else NHC's position
    carried along its motion), then on the eye: the point echo encloses
    best, averaged over 1-3 km, refined to the centre of its clear area.
    Where no eyewall encloses anything, it stays on the storm position.
    Michael at KEVX (10 Oct 2018): eye found 10 km from the first guess.
    Milton at KTBW (10 Oct 2024, 00:29Z), its eyewall open at 2-3 km: no
    eye claimed, so the volume stays on the storm position. A centre can
    also be clicked by hand.
  - NEXRAD scans can be listed for any past time (untick "ending now"), so
    a landfall can be studied after the fact.

Storm-relative velocity (Radar > NEXRAD and JMA)
  - A storm-relative velocity product: radial velocity less the storm
    motion's component along each ray, V - M cos(azimuth - heading). The
    motion comes from the recon fixes, NHC's advisory, the best track
    (NHC's, or JTWC's for the West Pacific), or typed in, and is shown in
    the panel title. It is gridded into the 3-D volume too.

Saved cases and replay (File > Save case, Open case)
  - A .r10case file holds a whole case: every HDOB, vortex message and TEMP
    DROP exactly as issued (decoded again when opened), NHC's best track,
    forecast and model tracks, the ADT list, your estimator's file with its
    column choices (ignored from 2.5.0 on), the TDR grids that were opened
    or measured (gzipped, 2-7 MB each; with three or more you are asked
    whether to keep them) and the TDR evolution measurements. Imagery
    stays in its public archives; the case records which scans were used.
  - Opening one puts up a replay bar: play at 1x to 900x or drag the
    slider, and every tab shows only what had arrived by then -- each
    message at the time NHC issued it, a TDR analysis when HRD's job file
    says it was made. "Imagery at this time" fetches the satellite loop
    for the replay clock.

Fixes
  - SHIPS shear direction. SHIPS's text product gives the direction the
    shear comes from; 2.2.1 read it as the direction it points. It is now
    turned around, and the lsdiag file's SDDC (a heading) is used when it
    is there. For Polo on 25 Sep, SHIPS gives 9 kt toward 246 deg (0-500
    km, vortex removed; 200-800 km: 13 kt toward 256) and the TDR inner-
    core shear 13-15 kt toward 237-260 deg: the two agree. 2.2.1's "nearly
    opposite" was this error.
  - Dropsondes issued by NHC's backup centre (KWBC instead of KNHC) were
    skipped: 81 of Isaias's 131 drops.
  - Gridded TDR wind fields failed to draw with matplotlib 3.9 or newer
    (matplotlib.cm.get_cmap was removed).

Tests
  - New offline tests for each addition: a synthetic moving vortex flown
    by a synthetic aircraft (storm-relative positions, RMW, radii in every
    quadrant), surface reductions and level flight, the timeline, ADT and
    TCFuse files, matching and scores, radar gridding against an analytic
    atmosphere and a real JMA cycle, storm-relative velocity, the eye
    finder, TDR pass metrics and their cache, case save/open/replay, and
    GUI runs through every new tab with the network refused.

----------------------------
What arrived in 2.2.1
----------------------------

Experimental section (home page, or Ctrl+4): 3D structure
  - A gridded airborne TDR analysis in three dimensions: reflectivity
    (20/35/45 dBZ) or wind-speed (64/96/137 kt) isosurfaces, the vortex
    column, and the shear vectors, in a view you can rotate (oblique, from
    south, from west, top). Follows whatever is open in Radar > Airborne
    TDR, or pick a mission here.
  - Vortex column: the centre at each height, as the centroid of positive
    vorticity near the centre one level down. A centre may move at most
    10 km between levels 0.5 km apart; beyond that the column stops, so
    scattered echo near the top of the data is not mistaken for the vortex.
    Tilt is the offset between two heights (default 2 and 8 km).
  - Inner-core shear: the difference between the mean wind at two heights
    within a disc (default 120 km), after removing the symmetric vortex at
    each height -- averaging raw winds lets a 70 m/s vortex leak through
    gaps in the radar's coverage and pose as shear. Uncertainty comes from
    a jackknife over eight 45-degree sectors, and a confidence grade from
    that spread and the coverage.
  - SHIPS for comparison: SHIPS's 850-200 hPa shear for the nearest
    cycle, for Atlantic and eastern/central Pacific storms. The storm ID is
    looked up from NRL's sector file. (2.3.0 corrects its direction.)
  - Consistency checks: where the reflectivity maximum (azimuthal
    wavenumber 1) and the tilt sit relative to each shear direction.
    Sheared storms usually put both downshear to downshear-left, so this
    shows which shear the storm's structure is responding to.
  - "Shear & tilt" tab: a hodograph of the vortex-removed mean wind by
    height (levels with under 25% coverage left out), the centres by
    height over low-level reflectivity with the shear vectors, and
    reflectivity by shear-relative quadrant for each shear reference.

  What it found for Hurricane Polo (25 Sep, five passes, two aircraft)
  - TDR inner-core shear 13-15 kt toward 237-260 deg, within +-2-5 deg
    from pass to pass. 2.2.1 reported SHIPS as nearly opposite (toward
    66-76 deg); that was a misreading of SHIPS's direction, which is where
    the shear comes from. Corrected in 2.3.0: SHIPS 9 kt toward 246 deg,
    in agreement with the TDR. Relative to the TDR shear, the reflectivity
    maximum is downshear-left and the tilt left of downshear.

  Checked on a synthetic storm with known answers: the tilt, a 250 deg
  shear despite a 90-degree gap in coverage, and a convective maximum 70
  deg left of downshear are all recovered (tests/test_tdr3d.py).

  Needs scikit-image (install_deps.py adds it).

----------------------------
What arrived in 2.2.0
----------------------------

JMA radar (Radar > JMA)
  - Japan Meteorological Agency's 20 C-band Doppler radars, from NICT's
    mirror (pawr.nict.go.jp/jmadata/JMA-PolarCoordsRadar). One TAR per
    10-minute cycle holds every radar: N5 reflectivity, N6 Doppler
    velocity. The mirror runs about two days behind real time; the tab
    finds its newest day by itself.
  - Typhoons come from NRL's GeoIPS sector file. They are listed nearest-
    radar first, and the nearest JMA radar within reach (400 km for
    reflectivity, 250 km for velocity) is chosen, with a note on which
    products reach the storm. A storm out of every radar's reach gets none,
    as with NEXRAD. Any radar can be picked instead.
  - Reflectivity and Doppler velocity in one or two panels, any elevation
    of the cycle, hover for values, range, bearing and beam height. Drawn
    by the same code as the NEXRAD tab, with the NWS reflectivity table.
  - Velocities arrive already unfolded by JMA's dual-PRF processing
    (+-70 m/s), so no dealiasing is needed.

Reading JMA's GRIB2 directly -- no conversion step
  - The files use three templates generic GRIB2 software does not
    implement: 3.50120 (azimuth-range grid), 4.51022 (radar product by
    elevation, with each ray's elevation and PRF) and 5.200 (run-length
    packing with level values). recon10s decodes them itself
    (recon10s_jmagrib.py), octet by octet from JMA's own format document,
    in about 0.1 s per radar.
  - Details that matter: negatives are sign-and-magnitude, not two's
    complement; each sweep starts at its own azimuth; in reflectivity,
    level 1 is "no echo"; in velocity, level 1 is a real 0 m/s -- checked
    against the same sweep's reflectivity -- and "nothing" is level 0.

LROSE
  - LROSE cannot read these files: in lrose-core-20260425 neither the Radx
    library behind RadxConvert (readers for NEXRAD, DORADE, CfRadial, ODIM,
    Sigmet, Gamic, Rapic, HRD and others) nor its grib2 library (standard
    templates only) knows JMA's templates. The only JMA code in LROSE reads
    satellite data.
  - So recon10s goes the other way: "Save as CfRadial" writes a cycle's
    decoded sweeps as CfRadial 1.4 netCDF, which LROSE reads. Tested with
    the real lrose-core-20260425 Ubuntu 24.04 package: RadxPrint reads the
    files, and RadxConvert turns them into ODIM HDF5, UF, DORADE and
    CfRadial 2; reflectivity and velocity come back through UF identical to
    recon10s's decoder. Py-ART reads the CfRadial back bit-for-bit.
    (RadxConvert keeps the field names DBZ and VEL; for ODIM's standard
    DBZH/VRADH, rename them in RadxConvert's parameter file.)
  - For native support, LROSE would need a Radx reader class for these
    templates; recon10s_jmagrib.py documents every field such a class
    would decode.

----------------------------
What arrived in 2.1.1
----------------------------

Airborne TDR probe
  - Hover over any panel for values; click to pin a reading, click again
    to release it. Plan view and cross-section: every field at that point
    (wind speed, tangential and radial wind, vertical velocity,
    reflectivity, vorticity), with distance and bearing from the centre,
    latitude/longitude and height. Azimuthal mean: the mean of the shown
    field at that radius and height, the mean tangential wind, and the RMW
    at that height. Winds follow the knots/m-s setting.

NWS reflectivity colour table
  - Reflectivity in the NEXRAD and Airborne TDR tabs (and TDR on the
    satellite map) uses the NWS default GR2Analyst table, bundled as
    scripts/palettes/NWS-Default-Color-Table.pal and read at start-up:
    smooth shading within each band, sharp steps at 10, 20, 35, 50, 65 and
    75 dBZ, nothing drawn below 0 dBZ. The colours follow GR2Analyst's
    rules exactly (checked every 0.01 dBZ); the file can be edited or
    swapped for another .pal.

----------------------------
What arrived in 2.1.0
----------------------------

A Radar section (home page, or Ctrl+3), with two tabs.

NEXRAD
  - Level II from the public archive (s3://unidata-nexrad-level2), for the
    WSR-88D nearest each active storm. Storms are listed nearest-radar
    first; the nearest site within reflectivity's reach (286 mi / 460 km)
    is chosen automatically, and the tab says whether velocity and
    dual-pol (about 186 mi) reach the centre too. Any of the 163 sites can
    be picked or typed instead.
  - Reflectivity, radial velocity, differential reflectivity and
    correlation coefficient, in 1, 2 or 4 panels, each panel's product
    chosen separately. Any elevation; scans from the last N hours with
    older/newer/latest, and "Follow new scans".
  - Velocity is dealiased (Py-ART region-based) by default: hurricane winds
    routinely exceed the ~47 kt Nyquist velocity of the low cuts.
  - Hover for the value in every panel, range and bearing from the radar,
    and beam height. Recon centre fixes within +-90 min are marked.
  - Built on MWSynth 0.169's unused NEXRAD scripts. The station table
    (163 sites, TDWR excluded) is regenerated from the same NCEI list with
    the same parser, and keeps its API and 200-mile default. MWSynth's
    reader had never been run on a real file; doing so showed a volume's
    low elevations are split cuts -- velocity lives in a second sweep at
    the same angle -- so "sweep 0", as MWSynth read it, has no velocity.
    Each product now takes the sweep that actually carries it.
  - Needs Py-ART: pip install arm_pyart (install_deps.py adds it).

Airborne TDR (gridded)
  - The files on NOAA's flight data server
    (seb.omao.noaa.gov/pub/flight/radar/<mission>/) are HRD's gridded tail
    Doppler radar analyses -- the numbers behind the TDR graphics:
      *_xy.nc.gz       a 3-D analysis per pass: storm-centred, 2 km grid
                       500 km across, 0-18 km every 0.5 km; wind speed,
                       tangential, radial and vertical wind, vorticity,
                       reflectivity, U and V, with lat/lon for every column
      *_vert_*.nc.gz   along-leg sections -- placeholders in the real-time
                       output (every value missing), though they do record
                       each leg's heading
      *.w.gz           the same analyses in HRD's older binary format
      *_radials.so.gz  radial-velocity super-observations for forecast
                       models' data assimilation
      *_analysis.tar   processing configuration
    Checked against HRD's own graphic for 260925I1 at 2211Z: the grid's 2 km
    maximum wind is 147.9 kt, exactly the value HRD printed.
  - Displayed four ways: plan view at any height (any field, wind vectors,
    RMW ring); vertical cross-section through the centre at any bearing,
    with one-click sections along the inbound and outbound legs; azimuthal
    mean (radius-height) with the RMW's tilt with height; and an overview
    of all three.
  - "Show on the satellite map" draws the plan view, georeferenced, over
    the Satellite loop. Checked against same-time GOES imagery of Polo:
    the TDR eye sits in the IR eye. The label says how far the analysis is
    from the frame on screen.
  - Classic netCDF, read with SciPy: no new dependency.

Packaging
  - The PyInstaller spec now bundles Py-ART with its config and colour
    tables. "recon10s --smoke-test" reports whether Py-ART, SciPy, cartopy
    and h5py load; with RECON10S_SMOKE_L2 set to a Level II file it also
    reads and dealiases it inside the packaged app.

----------------------------
What arrived in 2.0.0
----------------------------

One set of recon data, shared
  - Imagery > Satellite loads a storm's recon once -- NHC HDOBs, vortex
    messages, dropsondes, plus your converted IWG1 HDOB -- and every other
    tab uses it: Analysis, Microwave and the map.
  - "Load all recon for this storm" fetches all of it in one go.

Analysis (Reconnaissance > Analysis)
  - Centre fixes: NHC's vortex data messages (URNT12/URPN12), fully decoded
    -- position, minimum pressure, eye shape and diameter, inbound and
    outbound maximum winds with range, bearing and time.
  - Fixes derived from the obs for flights without vortex messages
    (converted NOAA IWG1 data): the lightest wind near the lowest height on
    each pass. Checked against all six of AF302/AF306's Nolo fixes on 26 Sep:
    within 0.8-3.3 km and under a minute of NHC's.
  - Radial profiles: flight-level wind against distance from the centre,
    inbound left of zero and outbound right, with extrapolated surface
    pressure below. Select several fixes to compare passes. Peak winds match
    the vortex messages to within 1-2 kt.
  - Tilt: surface centre relative to the flight-level centre, from a VDM
    "SFC CNTR" remark or a centre dropsonde's splash point (the source is
    always shown; a splash point also includes the sonde's drift).
  - Storm motion from a fit of the recent fixes.
  - Mission timeline: every flight's HDOB batches (coloured by peak wind),
    dropsondes and fixes on one time axis.

Satellite map
  - Overlays grouped into categories down the left: Recon obs, Fixes &
    dropsondes, Forecast, Lightning, Live & alerts, Tools.
  - Centre fixes (diamonds: vortex messages, crosses: derived) with minimum
    pressure, and a motion vector.
  - Peak flight-level wind labelled on each leg of each pass.
  - HDOB quality flags: winds flagged questionable are faded (or hidden, or
    shown normally). Obs with a flagged position are dropped -- the first
    line of a mission can read 0000N 00000E, which used to stretch "Zoom to
    recon" across half the globe.
  - NHC forecast cone and track, and the latest model cycle from the ATCF
    a-deck (official, consensus, global and hurricane models, each group
    switchable).
  - GLM lightning: flashes in the N minutes before each frame's scan,
    fetched for the frame on screen and cached.
  - More GOES imagery: bands 8, 10, 14 and 15, plus WV - IR (band 8 - 13;
    colour marks overshooting tops, above 0 C), Airmass RGB and Night
    Microphysics RGB, built from same-scan bands.
  - Measure: drag across the map for distance (km, nm) and bearing.
  - Export the loop as an animated GIF (or MP4 when ffmpeg is installed).
  - Auto-update: new scans join the loop and loaded recon, fixes, drops and
    forecast are fetched again every N minutes.
  - Alerts: central pressure at or below a value, a fall between fixes,
    flight-level wind above a value, or every new dropsonde -- a desktop
    notification, a beep, and a log.
  - The storm list fills in on its own at start.

Microwave
  - Recon barbs, dropsondes and centre fixes drawn on the passes, within
    +-90 minutes of the pass (adjustable).

TDR (Imagery > TDR)
  - NOAA P-3 tail Doppler radar analyses from HRD's real-time page, found
    by storm. The swath (reflectivity and wind speed at a height, 1-18 km on
    a slider) is shown above the azimuthal mean (radius-height section of
    reflectivity, tangential or radial wind), plus plan views, zoom, vortex
    tilt and TDR centre fixes. HRD publishes these as images, so they are
    shown as HRD draws them rather than on the satellite map. Double-click
    any image for full size.

Dropsondes
  - TEMP DROP parts C and D (above 100 hPa). These only exist for sondes
    released that high -- NASA Global Hawk or WB-57 -- since a G-IV
    releases near 175 hPa and its sondes are complete in parts A and B.

Sessions
  - The session (section, storm, imagery, layers, what was loaded) is saved
    on exit. The home page's "Resume last session" puts it back and fetches
    its data again.

Tests and packaging
  - tests/: pytest suite on real NHC messages from Hurricane Nolo -- HDOB,
    vortex message and TEMP DROP decoding, parts C/D, dating across 00Z,
    dated windows, derived fixes and profiles against the vortex message --
    plus a GUI smoke test (pytest-qt) that opens every section and feeds
    real data through the panels. All offline:
        pip install pytest pytest-qt
        python -m pytest tests
  - packaging/: PyInstaller spec and build scripts for a standalone app
    (build_windows.bat, build.sh). A packaged app keeps its settings and
    saved credentials in the user's config folder (%APPDATA%\recon10s,
    ~/.config/recon10s, ~/Library/Application Support/recon10s).
    "recon10s --smoke-test" starts it, opens every section and quits.

About SFMR
  - Left out, as asked. For the record, SFMR is still used operationally
    by NHC on both NOAA and USAF aircraft; it has known biases (heavy rain
    at low wind speeds, shallow water, a 2015 recalibration at high winds)
    rather than being unreliable as a whole.

----------------------------
What arrived in 1.7.1
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

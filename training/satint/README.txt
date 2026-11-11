Training the satellite intensity model
======================================

The app ships the trained network (scripts/models/satint_v1.npz), so nothing
here is needed to use Experimental > Satellite intensity. These scripts
rebuild it from public data, for anyone who wants to check or retrain it.
They need PyTorch (CPU is enough); the app itself does not. How well it does
is in the main README ("How good it is"); validation.png here draws it
against CIMSS D-PRINT and D-MINT on the same cases.

What it learns
  - Best-track maximum wind and central pressure (quantiles 10, 25, 50, 75
    and 90 %) from
      * 10.3-um IR on a 128 x 128 grid, 4 km apart, centred on the storm, at
        the estimate time and 3, 6, 9 and 12 hours before;
      * a 37/89-GHz imager pass from the 12 hours before, on a 64 x 64 grid
        8 km apart (37V, 37H, 89V, 89H and where the swath has data), with
        the IR from the time of the pass beside it, how old the pass is, the
        sensor's footprint and how much of the inner 200 km it covered;
      * latitude, translation speed, basin, local solar hour, and which of
        the earlier IR frames exist.
  - In training each sample gets its newest pass 40 % of the time and
    otherwise the newest pass in a random older age bin (0-2, 2-4, 4-7 or
    7-12 h), and no pass at all 30 % of the time, so the same network
    weighs an old pass against how the IR has changed since, and works from
    IR alone. Each sample is rotated by a random angle and shifted a few km.

Data
  - TC PRIMED v01r01 (NOAA NESDIS / CIRA), s3://noaa-nesdis-tcprimed-pds
    (https://noaa-nesdis-tcprimed-pds.s3.amazonaws.com/), every overpass
    file of Atlantic, east and central Pacific storms 2000-2025: 98,185
    files of 977 storms. Read with HTTP range requests -- only the IR, the
    37/89-GHz channels and the storm metadata of each file, about 127 GB in
    all -- and reduced to the grids above as they arrive (1.3 GB). Its IR
    is CIRA/RAMMB's tropical cyclone IR archive (4 km, McIDAS-remapped
    GOES), the same 10.3-um band the app reads from GOES-East and -West.
    Cite: Razin, M. N., C. J. Slocum, J. A. Knaff, P. J. Brown, and M. M.
    Bell, 2023: Tropical Cyclone Precipitation, Infrared, Microwave, and
    Environmental Dataset (TC PRIMED). Bull. Amer. Meteor. Soc., 104,
    E1980-E1998, https://doi.org/10.1175/BAMS-D-21-0052.1; the data:
    Razin, Slocum, Knaff and Brown, 2022, NOAA NCEI,
    https://doi.org/10.25921/dmy1-0595
  - GOES-18 and -19 ABI band 13 (NOAA's open buckets on AWS), through the
    app's own pipeline, for the 2025 and 2026 checks.
  - NHC b-decks for 2026 (ftp.nhc.noaa.gov/atcf/btk/), and CIMSS's ADT,
    D-PRINT and D-MINT histories for the comparison
    (tropic.ssec.wisc.edu/real-time/{adt,DPRINT,DMINT}/).

Steps (run from this folder; data/ is made here)
   1. python index_tcprimed.py data/index.json AL,EP,CP 2000 2025
   2. python fetch_tcprimed.py data/index.json data/storms 20 2000 2025
      (98,185 files; about 2 hours on a fast connection; resumable)
   3. python build_samples.py data/storms data/d
   4. python train_cnn.py data/d data/m.pt --epochs 30 --train-years 2000-2023 \
             --val-years 2024-2025 --channels-last --resume
      (an hour or two on two cores; --resume carries on after an interruption).
      The recipe was fixed before training: 2024-2025 are only watched, never
      used to choose anything, and the model is the last epoch, data/m_last.pt.
   5. python export_net.py data/m_last.pt data/m.npz
   6. CIMSS's estimates for the comparison, and the GOES checks:
      python fetch_cimss.py data/cimss AL052025 EP122026 ...   (storm IDs)
      python evaluate_satint.py data/m.npz data/d data/eval_heldout.json \
             --years 2024-2025 --cimss data/cimss
      python goes_pairs_2025.py data/d data/storms data/goes2025
      python cmp_goes_archive.py data/d data/goes2025 data/goes_vs_archive.png
      python bdeck_json.py <folder of 2026 b-decks> data/bdecks_2026.json
      python goes_frames_2026.py data/bdecks_2026.json data/goes2026
      python evaluate_satint.py data/m.npz data/d data/eval_goes.json --skip-archive \
             --goes25 data/goes2025 --goes26 data/goes2026 --bdecks data/bdecks_2026.json \
             --cimss data/cimss --adt
      Several networks trained alike (--seed 1, 2, ...) can go into one file,
      which the app then averages: export_net.py data/m_last.pt,data/m1_last.pt ...
   7. python make_info.py data/d 2000-2023 data/eval_heldout.json data/eval_goes.json data/info.json
      python export_net.py data/m_last.pt ../../scripts/models/satint_v1.npz data/info.json
      python plot_validation.py data/eval_heldout.json data/eval_goes.json data/validation.png

Outside the basins it learnt (2.6.0: JTWC's West Pacific, Southern
Hemisphere and North Indian Ocean, and Himawari-9)
  The model was not retrained. These scripts measure it on storms of JTWC's
  basins, none of which it saw, and fit the small mapping onto JTWC's best
  track the app applies there (recon10s_satint.JTWC_ADJUST). Southern storms
  are mirrored north-south as the app does (evaluate_wsi.py checks that its
  inputs and the app's agree exactly).
   8. python index_tcprimed.py data/index_wp.json WP 2020 2025
      python index_tcprimed.py data/index_sh_io.json SH,IO 2020 2025
   9. python fetch_tcprimed.py data/index_wp.json data/storms_wsi 20 2020 2025
      python fetch_tcprimed.py data/index_sh_io.json data/storms_wsi 20 2020 2025
      (43,200 files of 356 storms; about an hour)
  10. python build_samples.py data/storms_wsi data/wsi
      python build_samples.py data/storms_wp data/wp      (a folder holding only the WP storms' files)
      python fetch_cimss_wsi.py data/cimss_wsi data/index_wp.json data/index_sh_io.json --years 2020-2025
  11. python evaluate_wsi.py data/m_last.pt data/wsi data/eval_wsi.json --decide 2020-2023 \
             --report 2024-2025 --cimss data/cimss_wsi
      The basin input (0 Atlantic, 1 eastern Pacific) is chosen on 2020-2023 and the results reported
      on 2024-2025.
  12. python fit_jtwc_adjust.py data/eval_wsi.json data/wsi data/jtwc_adjust.json data/cimss_wsi --no-vmax-shift
      Fits the mapping on 2020-2023 only and scores it on 2024-2025 (without --no-vmax-shift it also
      fits a Vmax shift: better overall, worse from 34 kt up, so not used).
  13. The app's Himawari-9 pipeline against TC PRIMED's archive images (West Pacific, 2023-2025), and
      how the estimates fare when Himawari sees a storm obliquely (its 70-degree limit lifted for the
      test; the archive there is mostly Meteosat's, seen squarely):
      python himawari_pairs.py data/wp data/storms_wsi data/h9pairs 2023-2025 300
      python evaluate_h9_pairs.py data/m_last.pt data/wp data/h9pairs data/eval_h9_pairs.json --png data/h9.png
      python himawari_pairs.py data/wsi data/storms_wsi data/h9oblique 2023-2025 300 2 55-85
      python evaluate_oblique.py data/m_last.pt data/wsi data/h9oblique data/eval_oblique.json
  14. python plot_validation_wsi.py data/eval_wsi.json data/jtwc_adjust.json data/wsi validation_wsi.png
      python live_check_wsi.py data/live.json wp292026 sh012027      (live, through the app's code)

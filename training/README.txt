Training the synthetic TDR model
================================

(The satellite intensity model of 2.5.0 has its own folder: satint/README.txt.)

The app ships the trained model (scripts/models/syntdr_v1.npz, 2.7 MB), so
nothing here is needed to use Experimental > Synthetic TDR. These scripts
rebuild it from public data, for anyone who wants to check or retrain it.

What it learns
  - From TDR: the 3-D storm-relative tangential and radial wind, divided by
    Vm (the strongest azimuthal-mean tangential wind at 2 km), on nodes of
    r / R (R: the radius of Vm), azimuth from the storm motion, and height
    0.5-12 km. The mean structure is a regression on intensity, size, shear
    along and across the motion, translation speed and latitude; the
    departures from it are 60 EOFs (EM-PCA, which copes with the gaps every
    TDR analysis has) and the variance they leave.
  - From flight-level data (IWG1): how much to trust the prior against the
    obs (kappa) and the flight-level error, inflated for errors correlated
    along a leg (gamma); the intensity calibration; the uncertainty by r/R.
    Each of these was measured on storms or years the prior had not seen.
    Nothing from IWG1 is in the model file except those fitted numbers.

Data
  - TC-RADAR v3m (NOAA AOML/HRD), https://www.aoml.noaa.gov/ftp/pub/hrd/data/radar/level3/
    Read over HTTP range requests -- only the variables used, not the
    35 GB of files. HRD asks users of TC-RADAR to let its radar team lead,
    Dr. Paul Reasor (paul.reasor@noaa.gov), know the nature of the project,
    and publications to cite Fischer, M. S., R. F. Rogers, P. D. Reasor, and
    J. F. Gamache, 2022: An analysis of tropical cyclone vortex and
    convective characteristics in relation to storm intensity using a novel
    airborne Doppler radar database. Mon. Wea. Rev., 150, 2255-2278,
    https://doi.org/10.1175/MWR-D-21-0223.1
  - HRD's real-time TDR analyses, https://seb.omao.noaa.gov/pub/flight/radar/
  - NOAA P-3 IWG1 flight-level data,
    https://seb.omao.noaa.gov/pub/flight/aamps_ingest/iwg1/ -- marked as
    Controlled Unclassified Information (CUI): use it carefully and at your
    own risk. Only the minutes of each TDR pass are read (range requests).
  - NHC best tracks (b-decks: ftp.nhc.noaa.gov/atcf/btk/, and
    atcf/archive/<year>/ for past seasons) and SHIPS diagnostics
    (atcf/lsdiag/, current season only; TC-RADAR carries SHIPS for
    1997-2024).

Steps (run from this folder; data/ is made here; about 2 hours on two cores)
   1. python extract_tcradar.py data/tcradar
   2. python dataset.py data/tcradar data/norm_X.npy
   3. python fetch_realtime.py data/seb 2025 2026
   4. download the 2025-2026 b-decks into a folder, then
      python best_tracks.py <that folder> data/bdecks_2025_2026.json
   5. python realtime_dataset.py data/seb data/bdecks_2025_2026.json data/seb
   6. python iwg1_windows.py
      python fetch_iwg1.py data/iwg1_windows_tcradar.json data/iwg1 2
      python fetch_iwg1.py data/iwg1_windows_seb.json data/iwg1 2
   7. python tune_kappa_gamma.py                   (shipped: kappa 0.25, gamma 6)
   8. python cross_validate.py                     (5 folds by storm, 2018-2024)
   9. python train_tcradar_all.py
      python score_realtime.py data/model_tcradar_all_k60.npz data/seb_rows2.json
  10. python calibrate_intensity.py
  11. python calibrate_spread.py data/model_tcradar_all_k60.npz
      (its factors are the spread_by_rho list in train_final.py)
  12. python summarize_validation.py
  13. python train_final.py ../scripts/models/syntdr_v1.npz

The other files are the pieces those steps use: rangefile.py (an HTTP
range-request file for h5py), synmodel.py (the training), export.py (the
app's model format), realval.py, evaluate.py, evalcase.py and
realtime_eval.py (fitting the app's model to a pass's IWG1 and scoring it
against the pass's TDR analysis).

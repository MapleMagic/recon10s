RECON10s (IWG1 to HDOB Converter & Plotter)
=======================

This tool converts HDOB data to specified time intervals (e.g., 10s, 30s, 1min) 
and plots the results on a Mercator projection world map with wind barbs and MSLP labels.

----------------------------
Required Python Packages
----------------------------

- numpy
- matplotlib
- cartopy
- tkinter (usually included with Python installations)
- concurrent.futures (part of the Python standard library)

----------------------------
Installation Instructions
----------------------------

1. Install Python 3.9+ (recommended).
2. Install required packages with pip:

   pip install numpy matplotlib cartopy
   
   There is also a file called "install_deps.py" in case you don't feel like loading up Command Prompt.

3. Tkinter is bundled with most Python installations.  
   If missing, install separately depending on your OS:
   - Windows: Usually already included
   - macOS: Usually already included
   - Linux (Debian/Ubuntu): sudo apt-get install python3-tk

----------------------------
Usage
----------------------------

1. Run the GUI script:
   python recon_gui.py

2. Select (or save a new) .txt file to output IWG1 data converted into HDOB format.
3. Choose a time interval, output options, and plotting preferences.
4. If plotting is enabled, a PNG map will be generated alongside the processed .txt.

----------------------------
Notes
----------------------------

- Plots support dark mode for easier viewing.
- Output text files and plots can be filtered by UTC time range.
- Conversion is multithreaded for performance.
- There is an outputs folder that is just for making IWG1 to HDOB conversion outputs easier to find. However, you can put an output in any folder.
----------------------------
Getting Recon Data
----------------------------

NOAA/AOML has IWG1 (Aircraft Data) files in their backend in this directory. It is technically private though, as it's marked as Controlled Unclassified Information (CUI), so use it carefully and at your own risk:  
- https://seb.omao.noaa.gov/pub/flight/aamps_ingest/iwg1/
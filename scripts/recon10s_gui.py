#!/usr/bin/env python3
"""
recon10s_gui.py — Dark-mode GUI (full background + controls in dark theme)

Drop-in replacement for your GUI. Preserves all features:
- threaded conversion with progress spinner
- start/end UTC validation (green valid, white empty, salmon invalid)
- tooltips for start/end formats
- passes --start/--end to recon10s so HDOB output is filtered
- optional plotting (calls recon10s_plot which draws dark plots)
- "Plot existing HDOB..." button
"""
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import datetime as dt
import os
import re
import threading
import queue
from typing import Optional

import recon10s
import recon10s_plot

# ---- Dark theme colors ----
APP_BG = "#0f1112"        # main window background (very dark)
PANEL_BG = "#121417"      # panel / inner frame
ENTRY_BG = "#1b1d1f"      # entry background
ENTRY_FG = "#e8eef1"      # entry foreground
BTN_BG = "#1f2930"        # button background
BTN_ACTIVE = "#2b3940"    # button active background
BTN_FG = "#ffffff"        # button foreground
LABEL_FG = "#e9eef2"
VALID_BG = "#144f14"      # darker green for valid (kept subtle)
INVALID_BG = "#8b3b3b"    # darker salmon-like for invalid
TOOLTIP_BG = "#ffffe0"

# ---- validation regexes ----
_time_colon_re = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")
_time_plain_re = re.compile(r"^\s*(\d{4}|\d{6})\s*$")


def validate_time_string(s: str) -> bool:
    if s is None:
        return True
    s2 = s.strip()
    if s2 == "":
        return True
    m = _time_colon_re.match(s2)
    if m:
        hh = int(m.group(1)); mm = int(m.group(2)); ss = int(m.group(3)) if m.group(3) else 0
        return 0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60
    m2 = _time_plain_re.match(s2)
    if m2:
        token = m2.group(1)
        if len(token) == 4:
            hh = int(token[:2]); mm = int(token[2:4]); ss = 0
        else:
            hh = int(token[:2]); mm = int(token[2:4]); ss = int(token[4:6])
        return 0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60
    return False


# ---- tooltip helper ----
class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT, background=TOOLTIP_BG,
                         relief=tk.SOLID, borderwidth=1, font=("Segoe UI", "8"))
        label.pack(ipadx=4, ipady=2)

    def hide(self, _event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


# ---- GUI logic, worker queue ----
worker_queue: "queue.Queue[dict]" = queue.Queue()


def choose_local_iwg1():
    path = filedialog.askopenfilename(title="Select local IWG1 file", filetypes=[("Text files","*.txt"),("All files","*.*")])
    if path:
        path_var.set(path)


def choose_output_file():
    out = filedialog.asksaveasfilename(title="Choose output HDOB file", defaultextension=".txt", filetypes=[("Text files","*.txt"),("All files","*.*")])
    if out:
        out_var.set(out)


def choose_hdob_and_plot():
    hdob = filedialog.askopenfilename(title="Select HDOB file to plot", filetypes=[("Text files","*.txt"),("All files","*.*")])
    if not hdob:
        return
    start = start_var.get().strip() or None
    end = end_var.get().strip() or None
    show_legend = legend_var.get()
    try:
        parsed, filtered = compute_counts(hdob, start, end)
        status_var.set(f"Parsed {parsed} records — {filtered} records in window")
        if plot_var.get():
            recon10s_plot.main(hdob, start_utc=start, end_utc=end, show_legend=show_legend)
    except Exception as exc:
        messagebox.showerror("Plot error", f"Plot failed:\n{exc}")


def compute_counts(hdob_file: str, start_utc: Optional[str], end_utc: Optional[str]) -> tuple:
    times, times_tup, lats, lons, mslps, wind_dirs, wind_spds = recon10s_plot.parse_hdob_file(hdob_file)
    n_total = len(times)
    if n_total == 0:
        return 0, 0
    secs = [hh*3600 + mm*60 + ss for (hh, mm, ss) in times_tup]
    try:
        start_sec = recon10s_plot._time_input_to_seconds(start_utc) if getattr(recon10s_plot, "_time_input_to_seconds", None) else None
    except Exception:
        start_sec = None
    try:
        end_sec = recon10s_plot._time_input_to_seconds(end_utc) if getattr(recon10s_plot, "_time_input_to_seconds", None) else None
    except Exception:
        end_sec = None
    if start_sec is None and end_sec is None:
        return n_total, n_total
    count_in = 0
    for s in secs:
        keep = True
        if start_sec is not None and end_sec is not None:
            if start_sec <= end_sec:
                keep = (start_sec <= s <= end_sec)
            else:
                keep = (s >= start_sec or s <= end_sec)
        elif start_sec is not None:
            keep = (s >= start_sec)
        elif end_sec is not None:
            keep = (s <= end_sec)
        if keep:
            count_in += 1
    return n_total, count_in


def on_time_entry_change(*_args):
    s = start_var.get()
    e = end_var.get()
    s_ok = validate_time_string(s)
    e_ok = validate_time_string(e)
    # color entries: green if valid and non-empty, white if empty, salmon-like if invalid
    if s.strip() == "":
        start_entry.config(bg=ENTRY_BG, fg=ENTRY_FG)
    else:
        start_entry.config(bg=VALID_BG if s_ok else INVALID_BG, fg=ENTRY_FG)
    if e.strip() == "":
        end_entry.config(bg=ENTRY_BG, fg=ENTRY_FG)
    else:
        end_entry.config(bg=VALID_BG if e_ok else INVALID_BG, fg=ENTRY_FG)
    run_button.config(state="normal" if s_ok and e_ok else "disabled")


def start_conversion_worker():
    url = url_var.get().strip()
    path = path_var.get().strip()
    mission = mission_var.get().strip()
    storm_date = date_var.get().strip()
    interval = interval_var.get()
    out_file = out_var.get().strip()
    start = start_var.get().strip() or None
    end = end_var.get().strip() or None
    do_plot = plot_var.get()
    show_legend = legend_var.get()

    if not (url or path):
        messagebox.showerror("Input error", "Please provide either a URL or a local IWG1 file path.")
        return
    if not out_file:
        messagebox.showerror("Input error", "Please choose an Output HDOB file.")
        return
    if storm_date:
        try:
            dt.datetime.strptime(storm_date, "%Y%m%d")
        except Exception:
            messagebox.showerror("Input error", "Storm date must be YYYYMMDD (e.g., 20250815).")
            return
    try:
        interval_i = int(interval)
        if interval_i not in (10, 30, 60, 120):
            raise ValueError()
    except Exception:
        messagebox.showerror("Input error", "Interval must be one of: 10, 30, 60, 120 seconds.")
        return
    if not validate_time_string(start_var.get()) or not validate_time_string(end_var.get()):
        messagebox.showerror("Input error", "Start/End time entries are invalid. Use HH:MM, HHMM, or HH:MM:SS.")
        return

    argv = []
    if path:
        argv += ["--path", path]
    else:
        argv += ["--url", url]
    if mission:
        argv += ["--mission", mission]
    if storm_date:
        argv += ["--storm-date", storm_date]
    argv += ["--interval", str(interval_i)]
    argv += ["--out", out_file]
    if start:
        argv += ["--start", start]
    if end:
        argv += ["--end", end]

    _set_controls_state(False)
    status_var.set("Running conversion...")
    prog.start(10)

    t = threading.Thread(target=_worker_thread_target, args=(argv, out_file, do_plot, show_legend, start, end), daemon=True)
    t.start()
    root.after(200, _poll_worker_queue)


def _worker_thread_target(argv, out_file, do_plot, show_legend, start, end):
    result = {'success': False, 'rc': None, 'err': None, 'out_file': out_file, 'parsed': 0, 'filtered': 0}
    try:
        try:
            rc = recon10s.main(argv)
        except SystemExit as se:
            rc = getattr(se, "code", None)
        result['rc'] = rc
        if os.path.exists(out_file):
            try:
                parsed, filtered = compute_counts(out_file, start, end)
                result['parsed'] = parsed
                result['filtered'] = filtered
                result['success'] = True
            except Exception:
                result['success'] = True
        else:
            result['err'] = f"Output file not produced: {out_file}"
            result['success'] = False
    except Exception as exc:
        result['err'] = f"Exception during conversion: {exc}"
        result['success'] = False
    worker_queue.put(result)


def _poll_worker_queue():
    try:
        res = worker_queue.get_nowait()
    except queue.Empty:
        root.after(200, _poll_worker_queue)
        return
    _on_worker_done(res)


def _on_worker_done(result: dict):
    prog.stop()
    _set_controls_state(True)
    out_file = result.get('out_file')
    if result.get('success'):
        parsed = result.get('parsed', 0)
        filtered = result.get('filtered', 0)
        txt = ""
        if os.path.exists(out_file):
            try:
                with open(out_file, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
            except Exception as exc:
                txt = f"(Could not read output file: {exc})"
        else:
            txt = "(Output file missing)"
        hdob_text.delete("1.0", tk.END)
        hdob_text.insert(tk.END, txt)
        status_var.set(f"Parsed {parsed} records — {filtered} records in window")
        if plot_var.get() and os.path.exists(out_file):
            ask = messagebox.askyesno("Run plot?", "Conversion finished. Do you want to display the plot now?\n\nNote: plotting will open a matplotlib window and may block the GUI while the plot window is open.")
            if ask:
                try:
                    recon10s_plot.main(out_file, start_utc=start_var.get().strip() or None, end_utc=end_var.get().strip() or None, show_legend=legend_var.get())
                except Exception as exc:
                    messagebox.showerror("Plot error", f"Plotting failed:\n{exc}")
    else:
        err = result.get('err') or f"Converter returned rc={result.get('rc')}"
        status_var.set("Conversion failed.")
        messagebox.showerror("Conversion failed", err)


def _set_controls_state(enabled: bool):
    state = "normal" if enabled else "disabled"
    widgets = (url_entry, path_entry, path_browse_btn, mission_entry, date_entry, interval_menu, out_entry, out_browse_btn, start_entry, end_entry, run_button, plot_chk, legend_chk, plot_existing_btn)
    for w in widgets:
        try:
            w.config(state=state)
        except Exception:
            pass
    if not enabled:
        run_button.config(state="disabled")
    else:
        s_ok = validate_time_string(start_var.get())
        e_ok = validate_time_string(end_var.get())
        run_button.config(state="normal" if (s_ok and e_ok) else "disabled")


# ---- Build GUI ----
root = tk.Tk()
root.title("IWG1 → HDOB Converter — Dark UI")
root.configure(bg=APP_BG)

padx = 8; pady = 6

# Ttk style adjustments (use 'clam' for easier color control)
style = ttk.Style()
style.theme_use("clam")
# Frame style
style.configure("Dark.TFrame", background=PANEL_BG)
# Labels
style.configure("Dark.TLabel", background=PANEL_BG, foreground=LABEL_FG, font=("Segoe UI", 10))
# Buttons
style.configure("Dark.TButton", background=BTN_BG, foreground=BTN_FG, relief="flat", padding=6, font=("Segoe UI", 10))
style.map("Dark.TButton",
          background=[("active", BTN_ACTIVE), ("pressed", BTN_ACTIVE)],
          foreground=[("disabled", "#777777"), ("!disabled", BTN_FG)])
# Checkbutton style
style.configure("Dark.TCheckbutton", background=PANEL_BG, foreground=LABEL_FG, font=("Segoe UI", 10))
# Progressbar style
style.configure("Dark.Horizontal.TProgressbar", troughcolor=PANEL_BG, background="#3b82c4", bordercolor=PANEL_BG)

# Outer frame
frame = ttk.Frame(root, style="Dark.TFrame", padding=(12, 12, 12, 12))
frame.grid(sticky="nsew")
root.columnconfigure(0, weight=1)

# Row 0: URL
ttk.Label(frame, text="IWG1 URL:", style="Dark.TLabel").grid(row=0, column=0, sticky="e", padx=padx, pady=pady)
url_var = tk.StringVar()
url_entry = tk.Entry(frame, textvariable=url_var, width=60, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG)
url_entry.grid(row=0, column=1, columnspan=2, sticky="w", padx=padx, pady=pady)

# Row 1: Local file
ttk.Label(frame, text="Local IWG1 file:", style="Dark.TLabel").grid(row=1, column=0, sticky="e", padx=padx, pady=pady)
path_var = tk.StringVar()
path_entry = tk.Entry(frame, textvariable=path_var, width=45, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG)
path_entry.grid(row=1, column=1, sticky="w", padx=padx, pady=pady)
path_browse_btn = ttk.Button(frame, text="Browse...", style="Dark.TButton", command=choose_local_iwg1)
path_browse_btn.grid(row=1, column=2, sticky="w", padx=padx, pady=pady)

# Row 2: Mission
ttk.Label(frame, text="Mission ID:", style="Dark.TLabel").grid(row=2, column=0, sticky="e", padx=padx, pady=pady)
mission_var = tk.StringVar(value="Put a recon mission's name here.")
mission_entry = tk.Entry(frame, textvariable=mission_var, width=60, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG)
mission_entry.grid(row=2, column=1, columnspan=2, sticky="w", padx=padx, pady=pady)

# Row 3: Storm date
ttk.Label(frame, text="Storm date (YYYYMMDD):", style="Dark.TLabel").grid(row=3, column=0, sticky="e", padx=padx, pady=pady)
date_var = tk.StringVar(value=dt.date.today().strftime("%Y%m%d"))
date_entry = tk.Entry(frame, textvariable=date_var, width=20, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG)
date_entry.grid(row=3, column=1, sticky="w", padx=padx, pady=pady)

# Row 4: Interval
ttk.Label(frame, text="Interval (s):", style="Dark.TLabel").grid(row=4, column=0, sticky="e", padx=padx, pady=pady)
interval_var = tk.StringVar(value="30")
interval_menu = ttk.OptionMenu(frame, interval_var, "30", "10", "30", "60", "120")
interval_menu.grid(row=4, column=1, sticky="w", padx=padx, pady=pady)

# Row 5: Output file
ttk.Label(frame, text="Output HDOB file:", style="Dark.TLabel").grid(row=5, column=0, sticky="e", padx=padx, pady=pady)
out_var = tk.StringVar(value="")
out_entry = tk.Entry(frame, textvariable=out_var, width=45, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG)
out_entry.grid(row=5, column=1, sticky="w", padx=padx, pady=pady)
out_browse_btn = ttk.Button(frame, text="Choose...", style="Dark.TButton", command=choose_output_file)
out_browse_btn.grid(row=5, column=2, sticky="w", padx=padx, pady=pady)

# Row 6: Start UTC
ttk.Label(frame, text="Start UTC (HH:MM or HH:MM:SS):", style="Dark.TLabel").grid(row=6, column=0, sticky="e", padx=padx, pady=pady)
start_var = tk.StringVar(value="")
start_entry = tk.Entry(frame, textvariable=start_var, width=20, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG)
start_entry.grid(row=6, column=1, sticky="w", padx=padx, pady=pady)
ToolTip(start_entry, "Start time formats accepted:\nHH:MM  HH:MM:SS  HHMM  HHMMSS\n(UTC time-of-day)")

# Row 7: End UTC
ttk.Label(frame, text="End UTC (HH:MM or HH:MM:SS):", style="Dark.TLabel").grid(row=7, column=0, sticky="e", padx=padx, pady=pady)
end_var = tk.StringVar(value="")
end_entry = tk.Entry(frame, textvariable=end_var, width=20, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG)
end_entry.grid(row=7, column=1, sticky="w", padx=padx, pady=pady)
ToolTip(end_entry, "End time formats accepted:\nHH:MM  HH:MM:SS  HHMM  HHMMSS\n(UTC time-of-day)")

# Row 8: Plot toggles
plot_var = tk.BooleanVar(value=True)
legend_var = tk.BooleanVar(value=True)
plot_chk = ttk.Checkbutton(frame, text="Plot after conversion", variable=plot_var, style="Dark.TCheckbutton")
plot_chk.grid(row=8, column=1, sticky="w", padx=padx, pady=pady)
legend_chk = ttk.Checkbutton(frame, text="Show color legend", variable=legend_var, style="Dark.TCheckbutton")
legend_chk.grid(row=8, column=2, sticky="w", padx=padx, pady=pady)

# Row 9: Action buttons & progress
run_button = ttk.Button(frame, text="Run Conversion", style="Dark.TButton", command=start_conversion_worker)
run_button.grid(row=9, column=1, sticky="w", padx=padx, pady=pady)
plot_existing_btn = ttk.Button(frame, text="Plot existing HDOB...", style="Dark.TButton", command=choose_hdob_and_plot)
plot_existing_btn.grid(row=9, column=2, sticky="w", padx=padx, pady=pady)

prog = ttk.Progressbar(frame, mode="indeterminate", length=200, style="Dark.Horizontal.TProgressbar")
prog.grid(row=9, column=0, padx=padx, pady=pady)

# Row 10: HDOB output display
ttk.Label(frame, text="HDOB output:", style="Dark.TLabel").grid(row=10, column=0, sticky="ne", padx=padx, pady=pady)
hdob_text = scrolledtext.ScrolledText(frame, width=90, height=18, wrap=tk.WORD, bg="#0b0c0d", fg=ENTRY_FG, insertbackground=ENTRY_FG)
hdob_text.grid(row=10, column=1, columnspan=2, padx=padx, pady=pady)

# Row 11: Status label
status_var = tk.StringVar(value="Ready.")
status_label = ttk.Label(frame, textvariable=status_var, anchor="w", style="Dark.TLabel")
status_label.grid(row=11, column=1, columnspan=2, sticky="we", padx=padx, pady=pady)

# Bind validation
start_var.trace_add("write", on_time_entry_change)
end_var.trace_add("write", on_time_entry_change)
on_time_entry_change()

# Make things expand nicely
root.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)
root.resizable(True, True)

# Set initial widget focus for better UX
url_entry.focus_set()

root.mainloop()


from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import datetime as dt
import os
import re
import threading
import queue
import json
import sys
import subprocess
from typing import Optional

# local project modules (expected to exist in same folder)
import recon10s
import recon10s_plot

# SETTINGS file path (next to this script)
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "recon10s_settings.json")

# -------------------------
# Simple validators
# -------------------------
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


# -------------------------
# Defaults & runtime settings
# -------------------------
DEFAULTS = {
    "coord_format": "decimal",   # "decimal" or "dms"
    "gui_theme": "dark",         # "dark" or "light"
    "plot_theme": "dark"         # "dark" or "light"
}
settings = DEFAULTS.copy()

# worker queue used by background thread
worker_queue: "queue.Queue[dict]" = queue.Queue()

# -------------------------
# ToolTip helper
# -------------------------
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
        label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#ffffe0",
                         relief=tk.SOLID, borderwidth=1, font=("Segoe UI", "8"))
        label.pack(ipadx=4, ipady=2)

    def hide(self, _event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

# -------------------------
# Settings persistence
# -------------------------
def load_settings():
    global settings
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            for k in DEFAULTS:
                if k in obj:
                    settings[k] = obj[k]
        except Exception:
            print("Warning: could not read settings file; using defaults")
    else:
        # write defaults so file exists and users can open it
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(settings, fh, indent=2)
        except Exception:
            pass


def save_settings_to_file():
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    except Exception as exc:
        raise RuntimeError(f"Failed to write settings file: {exc}")


def reset_settings_file():
    """Reset on-disk settings and in-memory settings to defaults."""
    try:
        if os.path.exists(SETTINGS_FILE):
            os.remove(SETTINGS_FILE)
    except Exception:
        pass
    # restore in-memory defaults and write defaults file
    global settings
    settings = DEFAULTS.copy()
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    except Exception:
        pass


# -------------------------
# Restart helper
# -------------------------
def restart_program():
    """Restart the current Python program, replacing the process."""
    python = sys.executable
    os.execv(python, [python] + sys.argv)


# -------------------------
# Platform-open helper
# -------------------------
def open_settings_file_in_editor():
    """Open the settings JSON using the platform's default app."""
    if not os.path.exists(SETTINGS_FILE):
        # try to create it with defaults
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(settings, fh, indent=2)
        except Exception as exc:
            messagebox.showerror("Open failed", f"Could not create settings file: {exc}")
            return
    try:
        if sys.platform.startswith("win"):
            os.startfile(SETTINGS_FILE)
        elif sys.platform == "darwin":
            subprocess.call(["open", SETTINGS_FILE])
        else:
            subprocess.call(["xdg-open", SETTINGS_FILE])
    except Exception as exc:
        messagebox.showerror("Open failed", f"Could not open settings file: {exc}")


# -------------------------
# File dialogs & plotting helpers
# -------------------------
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
    coord_format = coord_var.get()
    plot_theme = plot_theme_var.get()
    try:
        parsed, filtered = compute_counts(hdob, start, end)
        status_var.set(f"Parsed {parsed} records — {filtered} records in window")
        if plot_var.get():
            recon10s_plot.main(hdob, start_utc=start, end_utc=end, show_legend=show_legend,
                               coord_format=coord_format, plot_theme=plot_theme, show_plot=True)
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


# -------------------------
# GUI validation & control
# -------------------------
def on_time_entry_change(*_args):
    s = start_var.get()
    e = end_var.get()
    s_ok = validate_time_string(s)
    e_ok = validate_time_string(e)
    theme = gui_themes[gui_theme_var.get()]
    if s.strip() == "":
        start_entry.config(bg=theme["ENTRY_BG"], fg=theme["ENTRY_FG"])
    else:
        start_entry.config(bg=theme["VALID_BG"] if s_ok else theme["INVALID_BG"], fg=theme["ENTRY_FG"])
    if e.strip() == "":
        end_entry.config(bg=theme["ENTRY_BG"], fg=theme["ENTRY_FG"])
    else:
        end_entry.config(bg=theme["VALID_BG"] if e_ok else theme["INVALID_BG"], fg=theme["ENTRY_FG"])
    run_button.config(state="normal" if (s_ok and e_ok) else "disabled")


def _set_controls_state(enabled: bool):
    state = "normal" if enabled else "disabled"
    widgets = (url_entry, path_entry, path_browse_btn, mission_entry, date_entry, interval_menu, out_entry, out_browse_btn,
               start_entry, end_entry, run_button, plot_chk, legend_chk, plot_existing_btn)
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


# -------------------------
# Worker thread / conversion
# -------------------------
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
    coord_format = coord_var.get()
    plot_theme = plot_theme_var.get()

    if not (url or path):
        messagebox.showerror("Input error", "Please provide either a URL or a local IWG1 file path.")
        return
    if not out_file:
        messagebox.showerror("Input error", "Please choose an output HDOB file.")
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
        messagebox.showerror("Input error", "Start/End time entries are invalid. Use HH:MM or HHMM format.")
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

    t = threading.Thread(target=_worker_thread_target, args=(argv, out_file, do_plot, show_legend, coord_format, plot_theme, start, end), daemon=True)
    t.start()
    root.after(200, _poll_worker_queue)


def _worker_thread_target(argv, out_file, do_plot, show_legend, coord_format, plot_theme, start, end):
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
            ask = messagebox.askyesno("Run plot?", "Conversion finished. Display plot now? (Plot opens a matplotlib window.)")
            if ask:
                try:
                    recon10s_plot.main(out_file,
                                      start_utc=start_var.get().strip() or None,
                                      end_utc=end_var.get().strip() or None,
                                      show_legend=legend_var.get(),
                                      coord_format=coord_var.get(),
                                      plot_theme=plot_theme_var.get(),
                                      show_plot=True)
                except Exception as exc:
                    messagebox.showerror("Plot error", f"Plotting failed:\n{exc}")
    else:
        err = result.get('err') or f"Converter returned rc={result.get('rc')}"
        status_var.set("Conversion failed.")
        messagebox.showerror("Conversion failed", err)


# -------------------------
# GUI themes small map for entries/text
# -------------------------
gui_themes = {
    "dark": {
        "APP_BG": "#0f1112", "PANEL_BG": "#121417", "ENTRY_BG": "#1b1d1f", "ENTRY_FG": "#e8eef1",
        "BTN_BG": "#1f2930", "BTN_FG": "#ffffff", "LABEL_FG": "#e9eef2",
        "VALID_BG": "#144f14", "INVALID_BG": "#8b3b3b", "TEXT_BG": "#0b0c0d"
    },
    "light": {
        "APP_BG": "#f6f6f6", "PANEL_BG": "#f0f0f0", "ENTRY_BG": "#ffffff", "ENTRY_FG": "#111111",
        "BTN_BG": "#e0e0e0", "BTN_FG": "#111111", "LABEL_FG": "#111111",
        "VALID_BG": "#c6efce", "INVALID_BG": "#f8d7da", "TEXT_BG": "#ffffff"
    }
}


def apply_gui_theme(theme_name: str):
    theme = gui_themes.get(theme_name, gui_themes["dark"])
    root.configure(bg=theme["APP_BG"])
    frame.configure(style="Main.TFrame")
    style.configure("Main.TFrame", background=theme["PANEL_BG"])
    style.configure("Main.TLabel", background=theme["PANEL_BG"], foreground=theme["LABEL_FG"])
    entries = (url_entry, path_entry, mission_entry, date_entry, out_entry, start_entry, end_entry)
    for e in entries:
        e.config(bg=theme["ENTRY_BG"], fg=theme["ENTRY_FG"], insertbackground=theme["ENTRY_FG"])
    hdob_text.config(bg=theme["TEXT_BG"], fg=theme["ENTRY_FG"], insertbackground=theme["ENTRY_FG"])
    on_time_entry_change()


# -------------------------
# Build GUI
# -------------------------
root = tk.Tk()
root.title("recon10s — IWG1 → HDOB Converter")
root.geometry("1050x760")

# Menu bar with File -> Show settings file
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Show settings file", command=open_settings_file_in_editor)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)

style = ttk.Style()
style.theme_use("clam")
style.configure("Main.TFrame", background="#121417")
style.configure("Main.TLabel", font=("Segoe UI", 10))
style.configure("Main.TButton", padding=6)
style.configure("Main.TCheckbutton", font=("Segoe UI", 10))
style.configure("Main.Horizontal.TProgressbar", thickness=10)

frame = ttk.Frame(root, style="Main.TFrame", padding=(12, 12, 12, 12))
frame.grid(sticky="nsew")
root.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)

notebook = ttk.Notebook(frame)
notebook.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(0, 8))

main_tab = ttk.Frame(notebook, style="Main.TFrame")
notebook.add(main_tab, text="Main")

settings_tab = ttk.Frame(notebook, style="Main.TFrame")
notebook.add(settings_tab, text="App Settings")

# ---------- Main tab ----------
padx = 8; pady = 6

ttk.Label(main_tab, text="IWG1 URL:", style="Main.TLabel").grid(row=0, column=0, sticky="e", padx=padx, pady=pady)
url_var = tk.StringVar()
url_entry = tk.Entry(main_tab, textvariable=url_var, width=72)
url_entry.grid(row=0, column=1, columnspan=2, sticky="w", padx=padx, pady=pady)

ttk.Label(main_tab, text="Local IWG1 file:", style="Main.TLabel").grid(row=1, column=0, sticky="e", padx=padx, pady=pady)
path_var = tk.StringVar()
path_entry = tk.Entry(main_tab, textvariable=path_var, width=52)
path_entry.grid(row=1, column=1, sticky="w", padx=padx, pady=pady)
path_browse_btn = ttk.Button(main_tab, text="Browse...", style="Main.TButton", command=choose_local_iwg1)
path_browse_btn.grid(row=1, column=2, sticky="w", padx=padx, pady=pady)

ttk.Label(main_tab, text="Mission ID:", style="Main.TLabel").grid(row=2, column=0, sticky="e", padx=padx, pady=pady)
mission_var = tk.StringVar(value="NOAA3 0605A ERIN")
mission_entry = tk.Entry(main_tab, textvariable=mission_var, width=72)
mission_entry.grid(row=2, column=1, columnspan=2, sticky="w", padx=padx, pady=pady)

ttk.Label(main_tab, text="Storm date (YYYYMMDD):", style="Main.TLabel").grid(row=3, column=0, sticky="e", padx=padx, pady=pady)
date_var = tk.StringVar(value=dt.date.today().strftime("%Y%m%d"))
date_entry = tk.Entry(main_tab, textvariable=date_var, width=20)
date_entry.grid(row=3, column=1, sticky="w", padx=padx, pady=pady)

ttk.Label(main_tab, text="Interval (s):", style="Main.TLabel").grid(row=4, column=0, sticky="e", padx=padx, pady=pady)
interval_var = tk.StringVar(value="30")
interval_menu = ttk.OptionMenu(main_tab, interval_var, "30", "10", "30", "60", "120")
interval_menu.grid(row=4, column=1, sticky="w", padx=padx, pady=pady)

ttk.Label(main_tab, text="Output HDOB file:", style="Main.TLabel").grid(row=5, column=0, sticky="e", padx=padx, pady=pady)
out_var = tk.StringVar(value="")
out_entry = tk.Entry(main_tab, textvariable=out_var, width=52)
out_entry.grid(row=5, column=1, sticky="w", padx=padx, pady=pady)
out_browse_btn = ttk.Button(main_tab, text="Choose...", style="Main.TButton", command=choose_output_file)
out_browse_btn.grid(row=5, column=2, sticky="w", padx=padx, pady=pady)

ttk.Label(main_tab, text="Start UTC (HH:MM or HHMM):", style="Main.TLabel").grid(row=6, column=0, sticky="e", padx=padx, pady=pady)
start_var = tk.StringVar(value="")
start_entry = tk.Entry(main_tab, textvariable=start_var, width=20)
start_entry.grid(row=6, column=1, sticky="w", padx=padx, pady=pady)

ttk.Label(main_tab, text="End UTC (HH:MM or HHMM):", style="Main.TLabel").grid(row=7, column=0, sticky="e", padx=padx, pady=pady)
end_var = tk.StringVar(value="")
end_entry = tk.Entry(main_tab, textvariable=end_var, width=20)
end_entry.grid(row=7, column=1, sticky="w", padx=padx, pady=pady)

plot_var = tk.BooleanVar(value=True)
legend_var = tk.BooleanVar(value=True)
plot_chk = ttk.Checkbutton(main_tab, text="Plot after conversion", variable=plot_var, style="Main.TCheckbutton")
plot_chk.grid(row=8, column=1, sticky="w", padx=padx, pady=pady)
legend_chk = ttk.Checkbutton(main_tab, text="Show color legend", variable=legend_var, style="Main.TCheckbutton")
legend_chk.grid(row=8, column=2, sticky="w", padx=padx, pady=pady)

run_button = ttk.Button(main_tab, text="Run Conversion", style="Main.TButton", command=start_conversion_worker)
run_button.grid(row=9, column=1, sticky="w", padx=padx, pady=pady)
plot_existing_btn = ttk.Button(main_tab, text="Plot existing HDOB...", style="Main.TButton", command=choose_hdob_and_plot)
plot_existing_btn.grid(row=9, column=2, sticky="w", padx=padx, pady=pady)

prog = ttk.Progressbar(main_tab, mode="indeterminate", length=200, style="Main.Horizontal.TProgressbar")
prog.grid(row=9, column=0, padx=padx, pady=pady)

ttk.Label(main_tab, text="HDOB output:", style="Main.TLabel").grid(row=10, column=0, sticky="ne", padx=padx, pady=pady)
hdob_text = scrolledtext.ScrolledText(main_tab, width=110, height=18, wrap=tk.WORD)
hdob_text.grid(row=10, column=1, columnspan=2, padx=padx, pady=pady)

status_var = tk.StringVar(value="Ready.")
status_label = ttk.Label(main_tab, textvariable=status_var, anchor="w", style="Main.TLabel")
status_label.grid(row=11, column=1, columnspan=2, sticky="we", padx=padx, pady=pady)

# -------------------------
# Settings tab widgets
# -------------------------
load_settings()  # load or write defaults before creating widgets that reflect them

coord_var = tk.StringVar(value=settings.get("coord_format", DEFAULTS["coord_format"]))
gui_theme_var = tk.StringVar(value=settings.get("gui_theme", DEFAULTS["gui_theme"]))
plot_theme_var = tk.StringVar(value=settings.get("plot_theme", DEFAULTS["plot_theme"]))

ttk.Label(settings_tab, text="Coordinate display:", style="Main.TLabel").grid(row=0, column=0, sticky="w", padx=padx, pady=pady)
coord_frame = ttk.Frame(settings_tab, style="Main.TFrame")
coord_frame.grid(row=0, column=1, sticky="w")
ttk.Radiobutton(coord_frame, text="Decimal degrees (e.g. 19.3457)", variable=coord_var, value="decimal").grid(row=0, column=0, sticky="w")
ttk.Radiobutton(coord_frame, text="Degrees/Minutes/Seconds (DMS)", variable=coord_var, value="dms").grid(row=1, column=0, sticky="w")
ttk.Label(settings_tab, text="(Affects how coordinates display on the plot)", style="Main.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", padx=padx, pady=(0,10))

ttk.Label(settings_tab, text="GUI theme:", style="Main.TLabel").grid(row=2, column=0, sticky="w", padx=padx, pady=pady)
gui_theme_frame = ttk.Frame(settings_tab, style="Main.TFrame")
gui_theme_frame.grid(row=2, column=1, sticky="w")
ttk.Radiobutton(gui_theme_frame, text="Dark (recommended)", variable=gui_theme_var, value="dark", command=lambda: on_gui_theme_change()).grid(row=0, column=0, sticky="w")
ttk.Radiobutton(gui_theme_frame, text="Light", variable=gui_theme_var, value="light", command=lambda: on_gui_theme_change()).grid(row=1, column=0, sticky="w")

ttk.Label(settings_tab, text="Plot theme:", style="Main.TLabel").grid(row=3, column=0, sticky="w", padx=padx, pady=pady)
plot_theme_frame = ttk.Frame(settings_tab, style="Main.TFrame")
plot_theme_frame.grid(row=3, column=1, sticky="w")
ttk.Radiobutton(plot_theme_frame, text="Dark (black background)", variable=plot_theme_var, value="dark").grid(row=0, column=0, sticky="w")
ttk.Radiobutton(plot_theme_frame, text="Light (white background)", variable=plot_theme_var, value="light").grid(row=1, column=0, sticky="w")
ttk.Label(settings_tab, text="(Plot theme applied the next time you request a plot)", style="Main.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", padx=padx, pady=(0, 10))

# Save / Reset / Reset+Restart buttons
save_btn = ttk.Button(settings_tab, text="Save Settings", style="Main.TButton")
save_btn.grid(row=5, column=1, sticky="w", padx=padx, pady=(12,6))
reset_btn = ttk.Button(settings_tab, text="Reset (confirm)", style="Main.TButton")
reset_btn.grid(row=5, column=2, sticky="w", padx=padx, pady=(12,6))
reset_restart_btn = ttk.Button(settings_tab, text="Reset + Restart", style="Main.TButton")
reset_restart_btn.grid(row=6, column=1, sticky="w", padx=padx, pady=(6,12))

ToolTip(save_btn, f"Settings file: {SETTINGS_FILE}")
ToolTip(reset_btn, f"Settings file: {SETTINGS_FILE}")
ToolTip(reset_restart_btn, f"Settings file: {SETTINGS_FILE}")

def on_save_settings():
    settings["coord_format"] = coord_var.get()
    settings["gui_theme"] = gui_theme_var.get()
    settings["plot_theme"] = plot_theme_var.get()
    try:
        save_settings_to_file()
    except Exception as exc:
        messagebox.showerror("Save error", f"Failed to save settings: {exc}")
        return
    # Offer restart (some theme bits require restart)
    if messagebox.askyesno("Restart required", "Settings saved. Restart GUI now to apply all changes?"):
        restart_program()
    else:
        messagebox.showinfo("Saved", "Settings saved. They will apply next time you start the GUI.")

def on_reset_settings():
    if not messagebox.askyesno("Confirm reset", "Reset settings to defaults? This will remove any custom settings."):
        return
    try:
        reset_settings_file()
    except Exception as exc:
        messagebox.showerror("Reset error", f"Failed to reset settings: {exc}")
        return
    if messagebox.askyesno("Restart required", "Settings reset to defaults. Restart GUI now to apply defaults?"):
        restart_program()
    else:
        messagebox.showinfo("Reset", "Settings reset to defaults. They will apply next time you start the GUI.")

def on_reset_and_restart():
    if not messagebox.askyesno("Confirm", "This will reset settings to defaults and immediately restart the GUI. Continue?"):
        return
    try:
        reset_settings_file()
    except Exception as exc:
        messagebox.showerror("Reset error", f"Failed to reset settings: {exc}")
        return
    # immediate restart
    restart_program()

save_btn.config(command=on_save_settings)
reset_btn.config(command=on_reset_settings)
reset_restart_btn.config(command=on_reset_and_restart)

# -------------------------
# Bindings & initial apply
# -------------------------
start_var.trace_add("write", on_time_entry_change)
end_var.trace_add("write", on_time_entry_change)

# Set widget states based on loaded settings
coord_var.set(settings.get("coord_format", DEFAULTS["coord_format"]))
gui_theme_var.set(settings.get("gui_theme", DEFAULTS["gui_theme"]))
plot_theme_var.set(settings.get("plot_theme", DEFAULTS["plot_theme"]))

# Entries need to exist for theme application
# (they were created above); apply GUI theme now
def on_gui_theme_change():
    apply_gui_theme(gui_theme_var.get())
    settings["gui_theme"] = gui_theme_var.get()

apply_gui_theme(gui_theme_var.get())

on_time_entry_change()
run_button.config(state="normal" if (validate_time_string(start_var.get()) and validate_time_string(end_var.get())) else "disabled")

# layout resizing
root.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)
main_tab.columnconfigure(1, weight=1)
settings_tab.columnconfigure(1, weight=1)
url_entry.focus_set()

root.mainloop()

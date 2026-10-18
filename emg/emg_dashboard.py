"""Live LSL dashboard. Install: python -m pip install pylsl matplotlib numpy scipy"""
import csv
import json
from pathlib import Path
import threading
import time
import queue
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pylsl import StreamInlet, resolve_byprop, resolve_streams, cf_string
from emg_filters import EMGFilter

BG, PANEL, TEXT, MUTED = '#eef2f6', '#ffffff', '#172e43', '#52677b'
NAVY, TEAL, BORDER = '#12283e', '#087f86', '#d5dfe8'
COLORS = ['#087f86', '#4167b2', '#a06419', '#476d67']

# Front-view sensor regions. The user assigns a channel to its physical placement.
BODY_REGIONS = {
    'Shoulder': [(34,63),(46,59),(59,69),(61,89),(49,97),(40,86)],
    'Upper arm': [(47,91),(62,92),(65,112),(62,128),(51,129),(46,112)],
    'Forearm': [(52,135),(63,134),(71,160),(76,190),(68,194),(59,166)],
    'Thigh': [(8,199),(26,196),(29,224),(24,249),(11,251),(7,225)],
    'Lower leg': [(12,263),(24,263),(23,283),(19,307),(10,309),(10,288)],
}


def intensity_color(level):
    """Visual green -> amber -> red scale, not a clinical threshold."""
    stops = [(62, 218, 145), (250, 197, 77), (249, 83, 99)]
    level = float(np.clip(level, 0, 1))
    a, b = (stops[0], stops[1]) if level < .5 else (stops[1], stops[2])
    fraction = level * 2 if level < .5 else (level - .5) * 2
    return '#' + ''.join(f'{round(x + (y-x)*fraction):02x}' for x, y in zip(a, b))


class BodyView(tk.Canvas):
    """Front-facing schematic; the wearer's right arm is on the viewer's left."""
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL, highlightthickness=0, width=340)
        self.level = None
        self.side = 'Right'
        self.region = 'Forearm'
        self.caption = 'Waiting for live data'
        self.bind('<Configure>', lambda _: self.draw())

    def show(self, level, side, caption, region='Forearm'):
        self.level, self.side, self.caption = level, side, caption
        self.region = region
        self.draw()

    def draw(self):
        self.delete('all')
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 20:
            return
        scale = min(w / 240, max(100, h-120) / 360)
        cx, top = w/2, 52
        def coords(points):
            return [v for x, y in points for v in (cx + x*scale, top + y*scale)]
        def shape(points, fill='#d7e2ed', outline='#98aec3', width=1):
            return self.create_polygon(coords(points), fill=fill, outline=outline,
                                       width=width, smooth=True, splinesteps=18)
        self.create_text(18, 22, text='Body map', anchor='w', fill=TEXT,
                         font=('Segoe UI', 10, 'bold'))
        self.create_text(w-16, 22, text='Front view', anchor='e', fill=MUTED, font=('Segoe UI', 8))
        # Head, neck, torso, pelvis, and legs.
        self.create_oval(cx-21*scale, top, cx+21*scale, top+45*scale,
                         fill='#d7e2ed', outline='#98aec3')
        shape([(-10,42), (10,42), (13,60), (-13,60)])
        shape([(-15,55),(-42,60),(-48,78),(-32,124),(-25,169),
               (25,169),(32,124),(48,78),(42,60),(15,55)])
        shape([(-25,160),(-30,192),(-15,206),(0,193),(15,206),(30,192),(25,160)])
        for sign in [-1, 1]:
            mirror = lambda points: [(sign*x,y) for x,y in points]
            shape(mirror([(5,197),(28,189),(32,233),(24,264),(22,302),(12,321),(4,313),(9,269),(4,231)]))
            shape(mirror([(12,310),(24,308),(34,324),(31,331),(8,330),(6,322)]))
            shape(mirror([(42,62),(59,68),(68,104),(65,133),(49,132),(43,103),(35,82)]))
            shape(mirror([(49,129),(65,128),(74,158),(80,195),(67,201),(55,169)]))
            shape(mirror([(67,196),(80,193),(87,212),(85,228),(77,230),(67,216)]))
        # Stylized torso contours, not a muscle/channel assignment.
        for sign in [-1,1]:
            shape([(sign*5,68),(sign*32,68),(sign*34,93),(sign*7,99)], '#c6d7e7', '#a3b9ce')
        self.create_line(*coords([(0,103),(0,154)]), fill='#a3b9ce', width=1)
        sign = -1 if self.side == 'Right' else 1
        points = BODY_REGIONS[self.region]
        target = [(sign*x,y) for x,y in points]
        color = intensity_color(self.level) if self.level is not None else '#53657a'
        if self.level is not None:
            # Layered outlines create a soft halo without an image dependency.
            shape(target, '#e0eee7', '#edf5f0', 14)
            shape(target, color, '#b9d1c7', 7)
        shape(target, color, color, 1)
        self.create_text(cx + sign*108*scale, top+np.mean([y for x,y in points])*scale,
                         text='R' if sign == -1 else 'L', fill=TEXT,
                         font=('Segoe UI', 12, 'bold'))
        bottom = h-44
        self.create_text(w/2, bottom-12, text=f'{self.side} {self.region.lower()}',
                         fill=TEXT, font=('Segoe UI', 9, 'bold'))
        self.create_text(w/2, bottom+10, text=self.caption, fill=TEXT,
                         font=('Segoe UI', 9))
        for i in range(100):
            self.create_rectangle(18+i*(w-36)/100, h-12, 18+(i+1)*(w-36)/100,
                                  h-7, fill=intensity_color(i/99), outline='')


def smooth_rms(times, values, seconds, gap_limit):
    """Trailing time-window RMS; restart at timestamp gaps, using available samples.

    No voltage conversion or clinical calibration is applied. Short segments use
    their available history. Nonfinite samples remain missing in the output.
    """
    result = np.full_like(values, np.nan, dtype=float)
    breaks = np.flatnonzero((np.diff(times) > gap_limit) | (np.diff(times) <= 0)) + 1
    for indices in np.split(np.arange(len(times)), breaks):
        if not len(indices):
            continue
        t, v = times[indices], values[indices]
        valid = np.isfinite(v)
        squares = np.vstack([np.zeros(v.shape[1]), np.cumsum(np.where(valid, v, 0)**2, axis=0)])
        counts = np.vstack([np.zeros(v.shape[1]), np.cumsum(valid, axis=0)])
        left = np.searchsorted(t, t - seconds, side='left')
        right = np.arange(1, len(t)+1)
        n = counts[right] - counts[left]
        mean = np.divide(squares[right]-squares[left], n,
                         out=np.full_like(v, np.nan, dtype=float), where=n > 0)
        result[indices] = np.sqrt(np.maximum(mean, 0))
        result[indices] = np.where(valid, result[indices], np.nan)
    return result


class Dashboard:
    def __init__(self, root, calibration_path=None):
        self.root = root
        root.title('Movement Studio | EMG dashboard')
        root.geometry(f'{min(1360, root.winfo_screenwidth()-80)}x{min(960, root.winfo_screenheight()-100)}')
        root.minsize(1100, 900)
        root.configure(bg=BG)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10), padding=(14,9))
        style.configure('TButton', background=PANEL, foreground=TEXT, borderwidth=0)
        style.map('TButton', background=[('active', '#dce8f3')],
                  foreground=[('disabled', '#718096')])
        style.configure('Accent.TButton', background=COLORS[0], foreground='#ffffff')
        style.map('Accent.TButton', background=[('active', '#06636a'), ('disabled', '#dce6f0')])
        style.configure('Invalid.TEntry', fieldbackground='#fff0ed', foreground='#99372a')
        style.configure('TEntry', fieldbackground=PANEL, foreground=TEXT, padding=10)
        style.configure('TCombobox', fieldbackground=PANEL, foreground=TEXT, padding=5)
        style.map('TCombobox', fieldbackground=[('readonly', PANEL)],
                  foreground=[('readonly', TEXT)])
        self.lock = threading.Lock()
        self.calibration_path = Path(calibration_path) if calibration_path else Path(__file__).with_name('emg_baselines.json')
        self.baselines = {}
        self.calibration = None
        self.baseline_error = ''
        try:
            if self.calibration_path.exists():
                records = json.loads(self.calibration_path.read_text(encoding='utf-8'))
                if not isinstance(records, dict):
                    raise ValueError('Expected saved baseline records')
                for key, record in records.items():
                    peak = float(record['peak'])
                    if not np.isfinite(peak) or peak <= 0:
                        raise ValueError('Invalid saved peak')
                    record['peak'] = int(peak) if peak.is_integer() else peak
                self.baselines = records
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.baseline_error = f'Cannot load baselines: {exc}'
        self.rows = deque(maxlen=100000)
        self.filtered_rows = deque(maxlen=100000)
        self.active_band = 'Off'
        self.filter_note = 'Filters apply on Connect.'
        self.filter_resets = 0
        self.visible_filter = 'Off'
        self.stop_event = threading.Event()
        self.worker = None
        self.status = 'Ready to connect. Start your sensor publisher first.'
        self.total = 0
        self.last_received = 0
        self.labels = []
        self.rate = 0
        self.axes = []
        self.lines = []
        self.visible_rows = []
        self.visible_labels = []
        self.visible_processed = []
        self.visible_plot_labels = []
        self.visible_mode = ''
        self.scale_mode = None
        self.render_key = None
        self.arrivals = deque()
        self.body_level = None
        self.body_value = None
        self.body_metric = 'Mean amplitude'
        self.empty_state = None
        self.stream_choices = {}
        self.discovery_queue = queue.Queue()
        self.scan_thread = None
        self.stream_type = 'emg'

        header = tk.Frame(root, bg=NAVY, padx=28, pady=15)
        header.pack(fill='x')
        tk.Label(header, text='Movement studio', fg='#ffffff', bg=NAVY,
                 font=('Segoe UI Semibold', 21)).pack(side='left')
        tk.Label(header, text='EMG signal workspace', fg='#c7d8e7', bg=NAVY,
                 font=('Segoe UI', 11)).pack(side='right')
        controls = tk.Frame(root, bg=BG)
        controls.pack(fill='x', padx=28, pady=(16,0))
        controls.columnconfigure(1, weight=1)
        tk.Label(controls, text='Stream', bg=BG, fg=MUTED,
                 font=('Segoe UI',10)).grid(row=0, column=0, padx=(0,12))
        self.name = tk.StringVar()
        self.stream_picker = ttk.Combobox(controls, textvariable=self.name, font=('Segoe UI', 10),
                                         state='readonly', width=20)
        self.stream_picker.grid(row=0, column=1, sticky='ew', padx=(0, 10))
        self.scan_button = ttk.Button(controls, text='Refresh', command=self.scan_streams)
        self.scan_button.grid(row=0, column=2, padx=4)
        self.connect_button = ttk.Button(controls, text='Connect', style='Accent.TButton', command=self.connect)
        self.connect_button.grid(row=0, column=3, padx=4)
        self.stop_button = ttk.Button(controls, text='Stop', command=self.stop, state='disabled')
        self.stop_button.grid(row=0, column=4, padx=4)
        self.save_button = ttk.Button(controls, text='Export view', command=self.save, state='disabled')
        self.save_button.grid(row=0, column=5, padx=4)
        self.discovery_label = tk.Label(root, text='Refresh to discover available sensor streams.',
                                         bg=BG, fg=MUTED, anchor='w', font=('Segoe UI',9))
        self.discovery_label.pack(fill='x', padx=28, pady=(5,0))
        self.stream_picker.bind('<<ComboboxSelected>>',
                                lambda _: self.discovery_label.configure(text=self.name.get()))

        self.status_label = tk.Label(root, text=self.status, bg=BG, fg=MUTED,
                                     anchor='w', font=('Segoe UI', 10), justify='left')
        self.status_label.pack(fill='x', padx=28, pady=(12, 12))
        self.content = tk.Frame(root, bg=BG)
        self.content.columnconfigure(0, minsize=340)
        self.content.columnconfigure(1, weight=1)
        self.content.rowconfigure(0, weight=1)
        self.chart_panel = tk.Frame(self.content, bg=PANEL, highlightbackground=BORDER,
                                    highlightcolor=BORDER, highlightthickness=1)
        self.chart_panel.grid(row=0, column=1, sticky='nsew')
        cards = tk.Frame(self.chart_panel, bg=PANEL)
        cards.pack(fill='x', padx=12, pady=(10, 8))
        self.card_values = []
        self.card_captions = []
        self.card_units = []
        for i, caption in enumerate(['Channel 1', 'Channel 2', 'Receiving', 'Samples']):
            cards.columnconfigure(i, weight=1, uniform='cards')
            card = tk.Frame(cards, bg=PANEL, padx=12, pady=8)
            card.grid(row=0, column=i, sticky='nsew', padx=(0 if i == 0 else 6, 0))
            heading = tk.Label(card, text=caption, bg=PANEL, fg=MUTED,
                               font=('Segoe UI', 9, 'bold'))
            heading.pack(anchor='w')
            self.card_captions.append(heading)
            value = tk.Label(card, text='—', bg=PANEL,
                             fg=COLORS[i] if i < 2 else TEXT,
                             font=('Segoe UI Semibold', 16))
            value.pack(anchor='w', pady=(5, 0))
            self.card_values.append(value)
            unit_text = tk.Label(card, text=['units pending', 'units pending', 'samples / second', 'samples'][i],
                                 bg=PANEL, fg=MUTED, font=('Segoe UI', 9))
            unit_text.pack(anchor='w')
            self.card_units.append(unit_text)
        self.metrics = tk.Label(self.chart_panel, text='Connect a sensor stream to begin.',
                                bg=PANEL, fg=MUTED, anchor='w', font=('Segoe UI', 9), wraplength=800)
        self.metrics.pack(fill='x', padx=18, pady=(0, 4))

        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=PANEL)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_panel)
        footer = tk.Frame(root, bg=BG)
        footer.pack(side='bottom', fill='x', padx=28, pady=12)
        settings = tk.Frame(footer, bg=BG)
        settings.pack(fill='x')
        self.mode = tk.StringVar(value='Combined amplitude')
        tk.Label(settings, text='Window (s)', fg=MUTED, bg=BG).pack(side='left')
        self.window = tk.StringVar(value='10')
        ttk.Combobox(settings, textvariable=self.window, values=['5', '10', '20'],
                     state='readonly', width=5).pack(side='left', padx=10)
        ttk.Button(settings, text='Fit scale', command=self.fit_scale).pack(side='left', padx=8)
        self.explanation = tk.Label(footer, text='Choose the body region where the selected sensor is placed.',
                                    fg=MUTED, bg=BG, anchor='w', font=('Segoe UI', 9), wraplength=1200, justify='left')
        self.explanation.pack(fill='x', pady=(8, 0))
        self.pause = tk.BooleanVar(value=False)
        tk.Checkbutton(settings, text='Freeze display', variable=self.pause,
                       bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                       activeforeground=TEXT).pack(side='right')
        target_controls = tk.Frame(footer, bg=BG)
        target_controls.pack(fill='x', before=settings, pady=(0,8))
        tk.Label(target_controls, text='Body mapping', bg=BG, fg=TEXT,
                 font=('Segoe UI',10,'bold'), width=13, anchor='w').pack(side='left')
        self.body_side = tk.StringVar(value='Right')
        ttk.Combobox(target_controls, textvariable=self.body_side, values=['Right','Left'],
                     state='readonly', width=7).pack(side='left', padx=8)
        self.body_region = tk.StringVar(value='Forearm')
        ttk.Combobox(target_controls, textvariable=self.body_region, values=list(BODY_REGIONS),
                     state='readonly', width=11).pack(side='left', padx=8)
        self.body_channel = tk.StringVar(value='Channel 1')
        self.channel_picker = ttk.Combobox(target_controls, textvariable=self.body_channel, values=['Channel 1','Channel 2'],
                     state='readonly', width=11)
        self.channel_picker.pack(side='left', padx=8)
        self.red_label = tk.Label(target_controls, text='Red at amplitude', bg=BG, fg=MUTED)
        self.red_label.pack(side='left', padx=(10,0))
        self.red_at = tk.StringVar(value='3000')
        self.red_entry = ttk.Entry(target_controls, textvariable=self.red_at, width=7)
        self.red_entry.pack(side='left', padx=8)
        tk.Label(target_controls, text='visual reference', bg=BG, fg=MUTED,
                 font=('Segoe UI',9)).pack(side='left')
        filter_controls = tk.Frame(footer, bg=BG)
        filter_controls.pack(fill='x', before=target_controls, pady=(0,8))
        tk.Label(filter_controls, text='EMG filters', bg=BG, fg=TEXT,
                 font=('Segoe UI',10,'bold'), width=13, anchor='w').pack(side='left')
        self.filter_band = tk.StringVar(value='20–350 Hz')
        self.filter_picker = ttk.Combobox(filter_controls, textvariable=self.filter_band,
                    values=['20–350 Hz', '20–500 Hz', 'Off'], state='readonly', width=14)
        self.filter_picker.pack(side='left', padx=8)
        tk.Label(filter_controls, text='+ 60 Hz notch (Q=30) · applies on Connect',
                 bg=BG, fg=MUTED).pack(side='left', padx=8)
        self.filter_label = tk.Label(footer, text=self.filter_note, bg=BG, fg=MUTED,
                                    anchor='w', justify='left', wraplength=1000, font=('Segoe UI',9))
        self.filter_label.pack(fill='x', before=self.explanation, pady=(6,0))
        calibration_controls = tk.Frame(footer, bg=BG)
        calibration_controls.pack(fill='x', before=settings, pady=(0,6))
        self.calibrate_button = ttk.Button(calibration_controls, text='Record baseline', command=self.start_calibration)
        self.calibrate_button.pack(side='left')
        self.save_peak_button = ttk.Button(calibration_controls, text='Save peak', command=self.save_calibration, state='disabled')
        self.save_peak_button.pack(side='left', padx=6)
        self.cancel_peak_button = ttk.Button(calibration_controls, text='Cancel', command=self.cancel_calibration, state='disabled')
        self.cancel_peak_button.pack(side='left')
        self.calibration_label = tk.Label(calibration_controls, text='', bg=BG, fg=MUTED,
                                         font=('Segoe UI',9), wraplength=550, justify='left')
        self.calibration_label.pack(side='left', padx=12)
        self.content.pack(fill='both', expand=True, padx=28)
        self.body = BodyView(self.content)
        self.body.grid(row=0, column=0, sticky='nsew', padx=(0,12))
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.show_empty('Ready to connect', 'Start your sensor publisher, select a stream from the dropdown,\nand select Connect. No readings have been received.')
        self.canvas.draw()
        self.body_side.trace_add('write', self.select_baseline)
        self.body_region.trace_add('write', self.select_baseline)
        self.mode.trace_add('write', self.select_baseline)
        self.select_baseline()
        root.bind('<Configure>', self.resize_layout)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.refresh)

    def resize_layout(self, event):
        if event.widget is not self.root:
            return
        self.status_label.configure(wraplength=max(400, event.width-64))
        self.discovery_label.configure(wraplength=max(400, event.width-64))
        self.explanation.configure(wraplength=max(400, event.width-64))
        self.filter_label.configure(wraplength=max(400, event.width-64))
        self.metrics.configure(wraplength=max(300, event.width-440))
        self.content.columnconfigure(0, minsize=300 if event.width < 1250 else 340)
        self.body.configure(width=288 if event.width < 1250 else 328)

    def show_empty(self, title, hint):
        state = (title, hint)
        if state == self.empty_state:
            return
        self.empty_state = state
        self.figure.clear()
        self.figure.text(.5, .58, title, ha='center', color=TEXT, fontsize=16)
        self.figure.text(.5, .43, hint, ha='center', va='center', color=MUTED, fontsize=10, linespacing=1.8)
        self.canvas.draw_idle()

    def scan_streams(self):
        if self.scan_thread and self.scan_thread.is_alive():
            return
        self.scan_button.configure(state='disabled')
        self.discovery_label.configure(text='Searching your local network…')
        def discover():
            try:
                self.discovery_queue.put((resolve_streams(wait_time=3), None))
            except Exception as exc:
                self.discovery_queue.put(([], str(exc)))
        self.scan_thread = threading.Thread(target=discover, daemon=True)
        self.scan_thread.start()

    def apply_discovery(self):
        try:
            streams, error = self.discovery_queue.get_nowait()
        except queue.Empty:
            return
        self.scan_button.configure(state='normal')
        # Keep exact outlet identity; never accidentally connect to a different
        # publisher just because two devices share a name.
        unique = {s.uid(): s for s in streams}
        choices = {}
        for s in sorted(unique.values(), key=lambda item: (item.name(), item.uid())):
            label = f'{(s.type() or "unknown").upper()} | {s.name()} | {s.channel_count()} ch | {s.nominal_srate():g} Hz'
            if label in choices:
                label += f' | {s.uid()[:8]}'
            choices[label] = s
        previous = self.stream_choices.get(self.name.get())
        self.stream_choices = choices
        self.stream_picker.configure(values=list(choices))
        selected = next((label for label, stream in choices.items()
                         if previous and stream.uid() == previous.uid()), None)
        if selected is None:
            selected = next((label for label, stream in choices.items() if stream.type().lower() == 'emg'),
                            next(iter(choices), ''))
        self.name.set(selected)
        self.discovery_label.configure(text=(f'Discovery failed: {error}' if error else
            f'{len(choices)} streams found. Stop acquisition before switching streams.' if choices else
            'No streams found. Start the sensor publisher, check your network, then Refresh.'))

    def connect(self):
        if self.worker and self.worker.is_alive():
            return
        self.cancel_calibration()
        name = self.name.get().strip()
        if not name:
            self.status_label.configure(text='Select an LSL stream. Use Refresh to discover devices.', fg='#99372a')
            return
        selected = self.stream_choices.get(name)
        if selected:
            self.stream_type = selected.type().lower()
            name = selected.name()
        self.stop_event.clear()
        self.active_band = self.filter_band.get() if self.stream_type == 'emg' else 'Off'
        with self.lock:
            self.rows.clear()
            self.filtered_rows.clear()
            self.filter_note = 'Waiting for stream sample rate…'
            self.filter_resets = 0
            self.total = 0
            self.last_received = 0
            self.labels = []
            self.arrivals.clear()
            self.status = 'Searching for stream…'
        self.pause.set(False)
        self.visible_rows = []
        self.visible_labels = []
        self.visible_processed = []
        self.scale_mode = None
        self.body_level = None
        self.body_value = None
        self.render_key = None
        self.axes, self.lines = [], []
        self.empty_state = None
        self.show_empty('Finding your sensor', 'Searching the local network. Keep the sensor publisher running.')
        self.canvas.draw_idle()
        for value in self.card_values:
            value.configure(text='—')
        for unit in self.card_units[:2]:
            unit.configure(text='units pending')
        self.metrics.configure(text='Waiting for stream metadata and samples.')
        self.worker = threading.Thread(target=self.receive, args=(name, selected), daemon=True)
        self.worker.start()

    def receive(self, name, selected=None):
        inlet = None
        try:
            streams = [selected] if selected is not None else resolve_byprop('name', name, timeout=5)
            if self.stop_event.is_set():
                return
            if not streams:
                raise RuntimeError('Stream not found. Check publisher and hotspot, then reconnect.')
            if streams[0].channel_format() == cf_string or streams[0].channel_count() < 1:
                raise RuntimeError('Choose a numeric sensor stream, not a text marker stream.')
            inlet = StreamInlet(streams[0], max_buflen=5)
            info = inlet.info(timeout=5)
            channel = info.desc().child('channels').child('channel')
            processor = None
            filter_note = 'Filters off · displaying unfiltered samples.'
            if self.active_band != 'Off':
                try:
                    high = 350 if self.active_band == '20–350 Hz' else 500
                    processor = EMGFilter(info.nominal_srate(), high=high)
                    filter_note = f'Band-pass {self.active_band} → 60 Hz notch (Q=30). Causal filtering; resets at gaps can cause transients.'
                except (ValueError, RuntimeError) as exc:
                    filter_note = f'Filters unavailable: {exc} Raw view remains available.'
            labels = []
            for i in range(info.channel_count()):
                labels.append((channel.child_value('label') or f'Channel {i+1}',
                               channel.child_value('unit') or 'device units — unverified'))
                channel = channel.next_sibling()
            with self.lock:
                self.labels = labels
                self.filter_note = filter_note
                self.rate = info.nominal_srate()
                self.status = 'Connected — waiting for samples'
            while not self.stop_event.is_set():
                samples, stamps = inlet.pull_chunk(timeout=.25, max_samples=4096)
                if stamps:
                    values = np.asarray(samples, dtype=float)
                    filtered = (processor.process(stamps, values) if processor else
                                values if self.active_band == 'Off' else np.full_like(values, np.nan))
                    with self.lock:
                        self.rows.extend(zip(stamps, values.tolist()))
                        self.filtered_rows.extend(filtered.tolist())
                        self.capture_calibration(filtered)
                        while self.rows and self.rows[0][0] < stamps[-1] - 21:
                            self.rows.popleft()
                            self.filtered_rows.popleft()
                        self.filter_resets = processor.resets if processor else 0
                        self.total += len(stamps)
                        self.last_received = time.monotonic()
                        self.arrivals.append((self.last_received, len(stamps)))
                        while self.arrivals and self.arrivals[0][0] < self.last_received - 5:
                            self.arrivals.popleft()
                        self.status = 'Live'
        except Exception as exc:
            with self.lock:
                self.status = f'Connection error: {exc}'
        finally:
            if inlet:
                inlet.close_stream()
            if self.stop_event.is_set():
                with self.lock:
                    self.status = 'Stopped'

    def refresh(self):
        self.apply_discovery()
        with self.lock:
            rows, labels = list(self.rows), list(self.labels)
            status, total, last, rate = self.status, self.total, self.last_received, self.rate
            arrivals = list(self.arrivals)
            filtered_rows = list(self.filtered_rows)
            filter_note, filter_resets = self.filter_note, self.filter_resets
        alive = self.worker is not None and self.worker.is_alive()
        with self.lock:
            calibration = dict(self.calibration) if self.calibration else None
        if calibration and (not alive or not last or time.monotonic()-last > 2):
            self.cancel_calibration('Recording cancelled: signal lost. Previous baseline kept.')
            calibration = None
        self.calibrate_button.configure(state='disabled' if calibration else 'normal')
        self.save_peak_button.configure(state='normal' if calibration else 'disabled')
        self.cancel_peak_button.configure(state='normal' if calibration else 'disabled')
        if calibration:
            self.calibration_label.configure(text=f'Recording {calibration["key"]} · peak {calibration["peak"]:,.2f} · {calibration["samples"]:,} samples')
        scanning = self.scan_thread is not None and self.scan_thread.is_alive()
        self.scan_button.configure(state='disabled' if alive or scanning else 'normal')
        self.connect_button.configure(state='disabled' if alive or scanning else 'normal')
        self.stream_picker.configure(state='disabled' if alive else 'readonly')
        self.filter_picker.configure(state='disabled' if alive else 'readonly')
        self.filter_label.configure(text=f'{filter_note}  ·  {filter_resets} resets' if filter_resets else filter_note)
        self.stop_button.configure(state='normal' if alive else 'disabled')
        self.save_button.configure(state='normal' if self.visible_rows else 'disabled')
        if alive and last and time.monotonic() - last > 2:
            status = f'No new samples for {time.monotonic()-last:.0f}s — checking connection'
        if not rows:
            if status.startswith('Connection error'):
                self.show_empty('Sensor unavailable', 'Check the stream name and publisher,\nthen select Connect to retry.')
            elif status == 'Stopped':
                self.show_empty('Acquisition stopped', 'Select Connect when you are ready to receive data.')
            elif status.startswith('Connected'):
                self.show_empty('Connected. Waiting for data.', 'The stream is visible, but no samples have arrived yet.')
        if self.pause.get():
            status += '  ·  Display frozen; acquisition continues'
        self.status_label.configure(text='●  ' + status,
                                    fg=TEAL if status == 'Live' else ('#99372a' if status.startswith('Connection error') else MUTED))
        now = time.monotonic()
        observed = sum(n for t, n in arrivals if t >= now - 5) / 5
        self.card_values[2].configure(text=f'{observed:,.0f} /s' if alive and last else '—')
        self.card_values[3].configure(text=f'{total:,}')
        key = (total, self.window.get(), tuple(labels), self.mode.get(), self.body_channel.get())
        if rows and not self.pause.get() and key != self.render_key:
            self.render_key = key
            self.channel_picker.configure(values=[f'Channel {i+1}' for i in range(len(labels))])
            mode = self.mode.get()
            times = np.array([r[0] for r in rows])
            raw = np.array([r[1] for r in rows], dtype=float)
            gap_limit = max(3 / rate, .02) if rate > 0 else .5
            source = (raw if self.active_band == 'Off' else
                      np.asarray(filtered_rows, dtype=float) if len(filtered_rows) == len(raw) else
                      np.full_like(raw, np.nan))
            plot_labels = labels
            combined = mode == 'Combined amplitude'
            combined_ready = (len(labels) >= 2 and labels[0][1] == labels[1][1]
                              and self.active_band != 'Off')
            if combined:
                processed = (np.mean(np.abs(source[:, :2]), axis=1, keepdims=True)
                             if combined_ready else np.full((len(raw), 1), np.nan))
                plot_labels = [('Amplitude analysis · channels 1 + 2',
                                labels[0][1] if combined_ready else 'unavailable')]
            elif mode == 'Raw':
                processed = raw
            elif mode == 'Filtered':
                processed = source if self.active_band != 'Off' else np.full_like(raw, np.nan)
            elif mode == 'Rectified':
                processed = np.abs(source)
            else:
                processed = smooth_rms(times, source, int(mode.split()[1])/1000, gap_limit)
            # Combined view drives the body from the same mean amplitude as the curve.
            # Other views retain their individually mapped channel's 200 ms RMS.
            self.channel_picker.configure(state='disabled' if combined else 'readonly')
            self.body_metric = 'Mean amplitude' if combined else 'RMS'
            self.red_label.configure(text='Red at amplitude' if combined else 'Red at RMS')
            if combined:
                self.body_value = float(processed[-1, 0])
            else:
                body_rms = smooth_rms(times, source, .2, gap_limit)
                channel_index = int(self.body_channel.get().split()[-1])-1
                self.body_value = (float(body_rms[-1, channel_index])
                                   if channel_index < body_rms.shape[1] else None)
            cutoff = rows[-1][0] - float(self.window.get())
            keep = times >= cutoff
            rows = [r for r in rows if r[0] >= cutoff]
            processed = processed[keep]
            self.visible_rows = rows
            self.visible_labels = labels
            self.visible_plot_labels = plot_labels
            self.visible_processed = processed.tolist()
            self.visible_mode = mode
            self.visible_filter = ('unfiltered' if self.active_band == 'Off' else filter_note)
            self.save_button.configure(state='normal')
            if len(self.axes) != len(plot_labels):
                self.empty_state = None
                self.figure.clear()
                self.axes = list(self.figure.subplots(len(plot_labels), 1, squeeze=False)[:, 0])
                self.lines = []
                for i, ax in enumerate(self.axes):
                    ax.set_facecolor(PANEL)
                    ax.tick_params(colors=MUTED, labelsize=9)
                    ax.grid(color='#b7c9d9', alpha=.45, linewidth=.6)
                    for spine in ax.spines.values():
                        spine.set_color('#c9d6e2')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    line, = ax.plot([], [], color=COLORS[i % len(COLORS)], linewidth=.9)
                    self.lines.append(line)
                self.figure.subplots_adjust(left=.09, right=.97, top=.80 if combined else .86, bottom=.26, hspace=1.1)
            x = np.array([r[0] for r in rows]) - rows[-1][0]
            y = np.array([r[1] for r in rows], dtype=float)
            # Break the line across gaps; never imply continuous measurements there.
            plot_y = processed.copy()
            plot_y[~np.isfinite(plot_y)] = np.nan
            gaps = np.flatnonzero(np.diff(x) > gap_limit) + 1
            plot_y[gaps] = np.nan
            for i, (ax, line) in enumerate(zip(self.axes, self.lines)):
                ax.set_title(plot_labels[i][0] if combined else f'{plot_labels[i][0]} · {mode}', loc='left', color=TEXT, fontsize=10)
                ax.set_title(plot_labels[i][1].replace('device units — unverified','unverified units'),
                             loc='right', color=MUTED, fontsize=8)
                line.set_data(x, plot_y[:, i])
                ax.set_xlim(-float(self.window.get()), 0)
                finite = plot_y[:, i][np.isfinite(plot_y[:, i])]
                if finite.size:
                    if mode not in ('Raw', 'Filtered'):
                        low, high = 0, max(float(finite.max()) * 1.15, 1e-9)
                    else:
                        pad = max(float(np.ptp(finite)) * .12, abs(float(finite.max())) * .05, 1e-9)
                        low, high = float(finite.min()) - pad, float(finite.max()) + pad
                    old_low, old_high = ax.get_ylim()
                    # Fit new peaks immediately; ease inward as peaks leave the visible window.
                    if self.scale_mode != mode:
                        ax.set_ylim(low, high)
                    else:
                        ax.set_ylim(low if low < old_low else old_low + .15*(low-old_low),
                                    high if high > old_high else old_high + .15*(high-old_high))
            self.scale_mode = mode
            self.axes[-1].set_xlabel('Seconds relative to latest sample', color=MUTED)
            for i in range(2):
                readouts = np.abs(source[keep]) if combined else processed
                value = readouts[-1, i] if i < readouts.shape[1] else np.nan
                self.card_values[i].configure(text=f'{value:,.1f}' if np.isfinite(value) else '—')
                self.card_captions[i].configure(text=f'Ch {i+1} amplitude' if combined else f'Channel {i+1} {mode.split()[0]}')
                self.card_units[i].configure(text=labels[i][1].replace('device units — unverified','unverified units') if i < len(labels) else 'not present')
            descriptions = {'Raw': 'Raw samples as received.',
                            'Combined amplitude': ('Mean amplitude = (|filtered channel 1| + |filtered channel 2|) / 2.' if combined_ready else
                                'Combined amplitude requires two channels with matching units and enabled filters. Enable filters and reconnect if needed.'),
                            'Filtered': 'Signed filtered EMG.' if self.active_band != 'Off' else 'Enable filters and reconnect to use Filtered view.',
                            'Rectified': 'Full-wave rectification: absolute value of the signal.'}
            self.explanation.configure(text=descriptions.get(mode, f'{mode} uses recent samples and restarts after gaps.') + ' '
                + ('Export includes raw channels and their mean amplitude.' if combined else
                   'Body location is assigned manually. Export saves raw and displayed values.'))
            self.metrics.configure(text=f'{len(labels)} channels  ·  {rate:g} Hz advertised  ·  '
                                   f'{len(gaps)} gaps > {gap_limit*1000:.0f} ms in view  ·  '
                                   'Receiving rate averaged over 5 seconds')
            self.canvas.draw_idle()
        self.update_body(alive, last)
        self.root.after(100, self.refresh)

    def update_body(self, alive, last):
        if self.stream_type != 'emg':
            self.body_level = None
            self.body.show(None, self.body_side.get(), 'Body intensity uses an EMG stream', self.body_region.get())
            return
        if not alive or not last or time.monotonic()-last > 2:
            self.body_level = None
            self.body.show(None, self.body_side.get(), 'No live data · highlight inactive', self.body_region.get())
            return
        try:
            upper = float(self.red_at.get())
            if not np.isfinite(upper) or upper <= 0:
                raise ValueError
        except ValueError:
            self.red_entry.configure(style='Invalid.TEntry')
            self.body.show(None, self.body_side.get(), 'Set a positive red reference value', self.body_region.get())
            return
        self.red_entry.configure(style='TEntry')
        if self.body_value is None or not np.isfinite(self.body_value):
            self.body_level = None
            self.body.show(None, self.body_side.get(), 'Signal unavailable', self.body_region.get())
            return
        target = float(np.clip(self.body_value / upper, 0, 1))
        if not self.pause.get():
            previous = 0.0 if self.body_level is None else self.body_level
            self.body_level = previous + .25*(target-previous)
        level = self.body_level
        caption = 'Frozen' if self.pause.get() else f'{self.body_metric} {self.body_value:,.0f}'
        self.body.show(level, self.body_side.get(), caption, self.body_region.get())

    def fit_scale(self):
        self.scale_mode = None
        self.render_key = None

    def baseline_key(self):
        return f'{self.body_side.get()} {self.body_region.get()}'

    def select_baseline(self, *_):
        self.cancel_calibration('Selection changed; recording cancelled.')
        record = self.baselines.get(self.baseline_key())
        combined = self.mode.get() == 'Combined amplitude'
        self.red_at.set(str(record['peak']) if record and combined else '3000')
        self.body_level = None
        self.calibration_label.configure(text=self.baseline_error or (
            f'{self.baseline_key()} · saved peak {record["peak"]:,}' if record and combined else
            'Use Combined amplitude to calibrate.' if not combined else
            f'{self.baseline_key()} · not calibrated (manual reference)'))

    def start_calibration(self):
        if self.baseline_error:
            messagebox.showerror('Baseline file unavailable', self.baseline_error)
            return
        with self.lock:
            if self.calibration is not None:
                return
            valid = (self.worker and self.worker.is_alive() and self.last_received
                     and time.monotonic()-self.last_received <= 2 and self.stream_type == 'emg'
                     and self.active_band != 'Off' and len(self.labels) >= 2
                     and self.labels[0][1] == self.labels[1][1] and self.filtered_rows
                     and np.all(np.isfinite(self.filtered_rows[-1][:2])))
            if valid and self.mode.get() == 'Combined amplitude':
                self.calibration = {'key': self.baseline_key(), 'peak': 0.0, 'samples': 0,
                                    'unit': self.labels[0][1], 'band': self.active_band,
                                    'sample_rate': self.rate, 'stream': self.name.get()}
            else:
                self.calibration_label.configure(text='Connect a live filtered EMG stream with two matching-unit channels; use Combined amplitude.')
                return
        self.pause.set(False)
        self.calibration_label.configure(text=f'Recording {self.baseline_key()} · flex, then select Save peak.')

    def capture_calibration(self, filtered):
        """Called under the acquisition lock; includes every sample in each chunk."""
        if self.calibration is None:
            return
        values = np.asarray(filtered, dtype=float)
        amplitudes = np.mean(np.abs(values[:, :2]), axis=1)
        amplitudes = amplitudes[np.isfinite(amplitudes)]
        if amplitudes.size:
            self.calibration['peak'] = max(self.calibration['peak'], float(amplitudes.max()))
            self.calibration['samples'] += len(amplitudes)

    def cancel_calibration(self, reason='Recording cancelled. Previous baseline kept.'):
        with self.lock:
            was_recording = self.calibration is not None
            self.calibration = None
        if was_recording:
            self.calibration_label.configure(text=reason)

    def save_calibration(self):
        with self.lock:
            if self.calibration is None:
                return
            if not self.last_received or time.monotonic()-self.last_received > 2:
                self.calibration_label.configure(text='No recent signal; cancel and reconnect before recording again.')
                return
            record = dict(self.calibration)
            if record['samples'] == 0 or record['peak'] <= 0:
                self.calibration_label.configure(text='No positive amplitude recorded yet. Keep recording or Cancel.')
                return
            # Positive amplitudes: round half upward, while retaining full precision during recording.
            record['peak'] = int(np.floor(record['peak'] + .5))
            if record['peak'] == 0:
                self.calibration_label.configure(text='Peak rounds to zero; a zero red threshold cannot be saved. Keep recording or Cancel.')
                return
            records = {**self.baselines, record['key']: record}
            try:
                temporary = self.calibration_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
                temporary.replace(self.calibration_path)
            except (OSError, ValueError) as exc:
                messagebox.showerror('Could not save baseline', str(exc))
                return
            self.baselines = records
            self.calibration = None
        self.red_at.set(str(record['peak']))
        self.body_level = None
        self.calibration_label.configure(text=f'{record["key"]} · saved peak {record["peak"]:,}')

    def stop(self):
        self.cancel_calibration()
        self.stop_event.set()
        with self.lock:
            self.status = 'Stopping…'

    def save(self):
        rows = getattr(self, 'visible_rows', [])
        if not rows:
            messagebox.showinfo('No data', 'Connect and receive samples first.')
            return
        path = filedialog.asksaveasfilename(defaultextension='.csv',
                    initialfile='emg_snapshot.csv', filetypes=[('CSV', '*.csv')])
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['lsl_timestamp_s'] + [f'{label} raw ({unit})' for label, unit in self.visible_labels]
                                    + [f'{label} {self.visible_mode} ({unit}) [{"unfiltered" if self.visible_mode == "Raw" else self.visible_filter}]' for label, unit in self.visible_plot_labels])
                    writer.writerows([stamp, *values, *displayed]
                                     for (stamp, values), displayed in zip(rows, self.visible_processed))
                messagebox.showinfo('Export complete', f'Saved {len(rows):,} samples.\n{path}')
            except OSError as exc:
                messagebox.showerror('Could not save', str(exc))

    def close(self):
        self.stop_event.set()
        self.root.destroy()


if __name__ == '__main__':
    # Keep text and Matplotlib crisp on Windows displays with scaling enabled.
    import sys
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    app = Dashboard(tk.Tk())
    app.root.after(100, app.scan_streams)
    app.root.mainloop()

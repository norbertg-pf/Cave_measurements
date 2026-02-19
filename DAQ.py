import sys
import threading
import nidaqmx.system
import numpy as np
import nidaqmx
from nidaqmx import stream_readers
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, LineGrouping, ProductCategory, ThermocoupleType, CJCSource
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QGridLayout,
                             QCheckBox, QListWidget, QListWidgetItem, QFileDialog, QScrollArea,
                             QTabWidget, QComboBox)
from PyQt5.QtCore import QTimer, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from nptdms import TdmsWriter, ChannelObject
from datetime import datetime
import collections
import time
from pathlib import Path
import DMM6510readout
import traceback

def get_terminal_name_with_dev_prefix(task: nidaqmx.Task, terminal_name: str) -> str:
    """Gets the terminal name with the device prefix."""
    for device in task.devices:
        if device.product_category not in [
            ProductCategory.C_SERIES_MODULE,
            ProductCategory.SCXI_MODULE,
        ]:
            return f"/{device.name}/{terminal_name}"
    raise RuntimeError("Suitable device not found in task.")

WRITE_CHANNEL = "Dev1/ao0"
DO_CHANNEL = "Dev1/port0/line0:1"
PROT_CHANNEL = "Dev1/ai1"


class NumericalIndicatorWidget(QWidget):
    """Custom widget for numerical readouts (RMS, P2P, Mean, Freq)."""
    def __init__(self, signals, remove_callback):
        super().__init__()
        self.remove_callback = remove_callback
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        top_layout = QHBoxLayout()
        self.signal_cb = QComboBox()
        self.signal_cb.addItems(signals)
        
        remove_btn = QPushButton("X")
        remove_btn.setFixedWidth(30)
        remove_btn.clicked.connect(lambda: self.remove_callback(self))
        
        top_layout.addWidget(self.signal_cb)
        top_layout.addWidget(remove_btn)
        
        self.type_cb = QComboBox()
        self.type_cb.addItems(["Current (100ms avg)", "RMS", "Peak-to-Peak", "Frequency"])
        
        self.value_label = QLabel("0.00")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #0055a4;")
        
        layout.addLayout(top_layout)
        layout.addWidget(self.type_cb)
        layout.addWidget(self.value_label)
        
        self.setStyleSheet("NumericalIndicatorWidget { border: 2px solid darkgray; border-radius: 6px; background-color: #f8f9fa;}")
        
    def calculate_and_update(self, data_chunk, rate, unit):
        if data_chunk is None or len(data_chunk) == 0:
            return
            
        calc_type = self.type_cb.currentText()
        val = 0.0
        display_unit = unit
        
        if calc_type == "Current (100ms avg)":
            samples = int(rate * 0.1)
            if samples > 0 and len(data_chunk) > samples:
                val = np.mean(data_chunk[-samples:])
            else:
                val = np.mean(data_chunk)
        elif calc_type == "RMS":
            val = np.sqrt(np.mean(np.square(data_chunk)))
        elif calc_type == "Peak-to-Peak":
            val = np.max(data_chunk) - np.min(data_chunk)
        elif calc_type == "Frequency":
            centered = data_chunk - np.mean(data_chunk)
            crossings = np.where((centered[:-1] < 0) & (centered[1:] >= 0))[0]
            if len(crossings) > 1:
                dt = (crossings[-1] - crossings[0]) / rate
                if dt > 0:
                    val = (len(crossings) - 1) / dt
            display_unit = "Hz"
            
        if calc_type == "Frequency":
            self.value_label.setText(f"{val:.2f} {display_unit}")
        else:
            self.value_label.setText(f"{val:.4g} {display_unit}")


class SubplotConfigWidget(QWidget):
    """Custom widget to handle the signal configuration for a single subplot."""
    def __init__(self, index, signals, remove_callback):
        super().__init__()
        self.remove_callback = remove_callback
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        top_layout = QHBoxLayout()
        self.title_label = QLabel(f"Subplot {index + 1}")
        self.title_label.setStyleSheet("font-weight: bold;")
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.on_remove)
        top_layout.addWidget(self.title_label)
        top_layout.addStretch()
        top_layout.addWidget(remove_btn)
        layout.addLayout(top_layout)
        
        grid = QGridLayout()
        self.checkboxes = {}
        row, col = 0, 0
        for sig in signals:
            cb = QCheckBox(sig)
            self.checkboxes[sig] = cb
            grid.addWidget(cb, row, col)
            col += 1
            if col > 2:  
                col = 0
                row += 1
        layout.addLayout(grid)
        
        self.setStyleSheet("SubplotConfigWidget { border: 1px solid gray; border-radius: 5px; }")
        
    def update_index(self, index):
        self.title_label.setText(f"Subplot {index + 1}")

    def on_remove(self):
        self.remove_callback(self)
        
    def get_selected_signals(self):
        return [sig for sig, cb in self.checkboxes.items() if cb.isChecked()]


class DAQControlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAQ Control GUI")

        self.full_resolution_buffer = None
        self.dmm_buffer = None
        self.dmmbuffer = []
        
        self.write_thread = None
        self.DMMread_thread = None
        self.read_thread = None
        self.plot_thread = None
        self.write_stop_flag = threading.Event()
        self.read_stop_flag = threading.Event()
        self.last_output_voltage = 0.0
        self.output_folder = ""
        self.write_task = None

        self.write_task_lock = threading.Lock()
        self.plot_ui_lock = threading.Lock() 

        self.sim_latest_voltage = 0.0
        self.start_timestamp = None
        self.threshold_triggered = False
        self.exported = False
        self.sample_nr = 0
        self.profile_data = []

        self.available_signals = ["AI0", "AI1", "AI2", "AI3", "AI4", "AI5", "DMM"]
        self.history_time = []
        self.history_data = {sig: [] for sig in self.available_signals}
        self.active_channel_configs = [] 

        self.indicator_widgets = []
        self.indicator_timer = QTimer()
        self.indicator_timer.timeout.connect(self.refresh_indicators)

        # --- TABS SETUP ---
        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        self.config_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Main Control")
        self.tabs.addTab(self.config_tab, "Channel Config")
        
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.tabs)
        self.setLayout(outer_layout)

        self.setup_main_tab()
        self.setup_config_tab()

        try:
            nidaqmx.system.Device('Dev1').reset_device()
            do_init = nidaqmx.Task()
            do_init.do_channels.add_do_chan(DO_CHANNEL, line_grouping=LineGrouping.CHAN_PER_LINE)
            do_init.write([False, False])
            do_init.close()
        except Exception as e:
            print(f"[WARN] DAQ Hardware not found at startup: {e}. Check 'Simulate Mode' to run anyway.")

    def setup_main_tab(self):
        main_layout = QHBoxLayout(self.main_tab)
        left_panel = QVBoxLayout()
        center_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        # --- LEFT PANEL: Numerical Indicators ---
        indicator_header = QLabel("<b>Numerical Indicators</b>")
        left_panel.addWidget(indicator_header)
        
        self.indicator_scroll_area = QScrollArea()
        self.indicator_scroll_widget = QWidget()
        self.indicator_scroll_layout = QVBoxLayout(self.indicator_scroll_widget)
        self.indicator_scroll_layout.addStretch()
        self.indicator_scroll_area.setWidget(self.indicator_scroll_widget)
        self.indicator_scroll_area.setWidgetResizable(True)
        self.indicator_scroll_area.setMinimumWidth(220)
        
        self.add_indicator_btn = QPushButton("Add Indicator")
        self.add_indicator_btn.clicked.connect(self.add_indicator)
        
        left_panel.addWidget(self.indicator_scroll_area)
        left_panel.addWidget(self.add_indicator_btn)


        # --- RIGHT PANEL: Controls ---
        self.export_button = QPushButton("Export Last 30s Data")
        self.export_button.clicked.connect(self.export_high_res_data)
        right_panel.addWidget(self.export_button)

        controls_layout = QGridLayout()
        self.write_rate_input = QLineEdit("1000")
        self.read_rate_input = QLineEdit("10000")
        self.average_samples_input = QLineEdit("100")
        self.plot_window_input = QLineEdit("10") # <--- NEW FIELD FOR PLOT TIME WINDOW
        self.threshold_input = QLineEdit("0.0002")
        self.averaged_filename_input = QLineEdit("test")
        self.simulate_checkbox = QCheckBox("Simulate Mode")

        self.write_active_label = QLabel("Write Active")
        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")
        self.shutdown_label = QLabel("Status: OK")
        self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")

        choose_folder_btn = QPushButton("Choose Output Folder")
        choose_folder_btn.clicked.connect(self.select_output_folder)
        self.folder_display = QLabel()
        
        controls_layout.addWidget(QLabel("Write Rate (Hz):"), 0, 0)
        controls_layout.addWidget(self.write_rate_input, 0, 1)
        controls_layout.addWidget(QLabel("Read Rate (Hz):"), 1, 0)
        controls_layout.addWidget(self.read_rate_input, 1, 1)
        controls_layout.addWidget(QLabel("Samples to Average Over:"), 2, 0)
        controls_layout.addWidget(self.average_samples_input, 2, 1)
        controls_layout.addWidget(QLabel("Plot View Window (s, 0=All):"), 3, 0)
        controls_layout.addWidget(self.plot_window_input, 3, 1)
        controls_layout.addWidget(QLabel("Voltage Threshold (V):"), 4, 0)
        controls_layout.addWidget(self.threshold_input, 4, 1)
        controls_layout.addWidget(QLabel("Averaged Data Filename:"), 5, 0)
        controls_layout.addWidget(self.averaged_filename_input, 5, 1)
        controls_layout.addWidget(choose_folder_btn, 6, 0, 1, 2)
        controls_layout.addWidget(QLabel("Keithley DMM IP"), 7, 0)
        self.Keithley_DMM_IP = QLineEdit("169.254.169.37")
        controls_layout.addWidget(self.Keithley_DMM_IP, 7, 1)
        controls_layout.addWidget(self.folder_display, 8, 0, 1, 2)
        controls_layout.addWidget(self.write_active_label, 9, 0, 1, 2)
        controls_layout.addWidget(self.simulate_checkbox, 10, 0, 1, 2)
        controls_layout.addWidget(self.shutdown_label, 11, 0, 1, 2)
        
        self.output_folder = r"\data"
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.folder_display.setText(f"Output Folder: {self.output_folder if len(self.output_folder)<40 else self.output_folder[-40:]}")

        # --- Dynamic Subplot Config (Right Panel) ---
        plot_control_layout = QVBoxLayout()
        plot_control_layout.addWidget(QLabel("<b>Dynamic Plot Configuration</b>"))
        
        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_widget = QWidget()
        self.plot_scroll_layout = QVBoxLayout(self.plot_scroll_widget)
        self.plot_scroll_layout.addStretch()
        self.plot_scroll_area.setWidget(self.plot_scroll_widget)
        self.plot_scroll_area.setWidgetResizable(True)
        
        self.add_subplot_btn = QPushButton("Add New Subplot")
        self.add_subplot_btn.clicked.connect(lambda: self.add_subplot())
        
        plot_control_layout.addWidget(self.plot_scroll_area)
        plot_control_layout.addWidget(self.add_subplot_btn)


        # --- CENTER PANEL: Plots & Main Buttons ---
        self.start_write_btn = QPushButton("Start Write")
        self.stop_write_btn = QPushButton("Stop Write")
        self.start_read_btn = QPushButton("Start Read")
        self.stop_read_btn = QPushButton("Stop Read")
        self.exit_btn = QPushButton("Exit")
        self.reset_btn = QPushButton("Reset All")

        self.start_write_btn.clicked.connect(self.start_write)
        self.stop_write_btn.clicked.connect(self.stop_write)
        self.start_read_btn.clicked.connect(self.start_read)
        self.stop_read_btn.clicked.connect(self.stop_read)
        self.exit_btn.clicked.connect(self.exit_application)
        self.reset_btn.clicked.connect(self.reset_all)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_write_btn)
        btn_layout.addWidget(self.stop_write_btn)
        btn_layout.addWidget(self.start_read_btn)
        btn_layout.addWidget(self.stop_read_btn)
        btn_layout.addWidget(self.exit_btn)
        btn_layout.addWidget(self.reset_btn)

        self.figure = plt.figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)
        self.axs = []
        self.subplot_widgets = []

        right_panel.addLayout(controls_layout)
        right_panel.addLayout(plot_control_layout) 
        
        center_panel.addWidget(self.canvas)
        center_panel.addLayout(btn_layout)

        # Assemble Main Tab
        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(center_panel, stretch=4)
        main_layout.addLayout(right_panel, stretch=1)

        self.add_subplot(["AI0", "AI1"])
        self.add_subplot(["AI2", "AI3", "AI4", "AI5"])
        self.add_subplot(["DMM"])

        # Default Indicators
        self.add_indicator("AI0", "RMS")
        self.add_indicator("AI1", "Frequency")

    def setup_config_tab(self):
        """Builds the UI for configuring channels with scaling, units, offsets, and CJC."""
        layout = QVBoxLayout(self.config_tab)
        
        grid = QGridLayout()
        grid.addWidget(QLabel("<b>Channel</b>"), 0, 0)
        grid.addWidget(QLabel("<b>Terminal Config</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Voltage Range</b>"), 0, 2)
        grid.addWidget(QLabel("<b>Sensor Type</b>"), 0, 3)
        grid.addWidget(QLabel("<b>Scale Factor</b>"), 0, 4)
        grid.addWidget(QLabel("<b>Unit</b>"), 0, 5)
        grid.addWidget(QLabel("<b>Offset (V)</b>"), 0, 6)

        self.channel_ui_configs = []
        term_options = ["RSE", "NRSE", "DIFF", "PSEUDO_DIFF"]
        range_options = ["-0.2 to 0.2", "-2.5 to 2.5", "-5 to 5", "-10 to 10"]
        sensor_options = ["None", "Type K"]

        def make_sensor_callback(sensor_cb, scale_input, unit_input, offset_input):
            def callback(index):
                sensor_type = sensor_cb.currentText()
                if sensor_type != "None":
                    scale_input.setText("1.0")
                    scale_input.setEnabled(False)
                    offset_input.setText("0.0")
                    offset_input.setEnabled(False)
                    unit_input.setText("°C")
                else:
                    scale_input.setEnabled(True)
                    offset_input.setEnabled(True)
                    unit_input.setText("V")
            return callback

        for i in range(6):
            ch_label = QLabel(f"Dev1/ai{i} (AI{i})")
            term_cb = QComboBox()
            term_cb.addItems(term_options)
            range_cb = QComboBox()
            range_cb.addItems(range_options)
            sensor_cb = QComboBox()
            sensor_cb.addItems(sensor_options)
            scale_input = QLineEdit("1.0")
            unit_input = QLineEdit("V")
            offset_input = QLineEdit("0.0")

            sensor_cb.currentIndexChanged.connect(make_sensor_callback(sensor_cb, scale_input, unit_input, offset_input))

            if i == 0:
                term_cb.setCurrentText("RSE")
                range_cb.setCurrentText("-10 to 10")
            elif i in [2, 3, 4]:
                term_cb.setCurrentText("DIFF")
                range_cb.setCurrentText("-0.2 to 0.2")
                sensor_cb.setCurrentText("Type K") 
                
                scale_input.setText("1.0")
                scale_input.setEnabled(False)
                offset_input.setText("0.0")
                offset_input.setEnabled(False)
                unit_input.setText("°C")
            else: 
                term_cb.setCurrentText("DIFF")
                range_cb.setCurrentText("-10 to 10")

            grid.addWidget(ch_label, i+1, 0)
            grid.addWidget(term_cb, i+1, 1)
            grid.addWidget(range_cb, i+1, 2)
            grid.addWidget(sensor_cb, i+1, 3)
            grid.addWidget(scale_input, i+1, 4)
            grid.addWidget(unit_input, i+1, 5)
            grid.addWidget(offset_input, i+1, 6)

            self.channel_ui_configs.append({
                "name": f"AI{i}",
                "terminal_str": f"Dev1/ai{i}",
                "term_cb": term_cb,
                "range_cb": range_cb,
                "sensor_cb": sensor_cb,
                "scale_input": scale_input,
                "unit_input": unit_input,
                "offset_input": offset_input
            })
            
        layout.addLayout(grid)
        
        self.measure_offsets_btn = QPushButton("Measure Offsets (1s)")
        self.measure_offsets_btn.clicked.connect(self.measure_ui_offsets)
        layout.addWidget(self.measure_offsets_btn)
        
        layout.addStretch()

    def measure_ui_offsets(self):
        configs = self.get_current_channel_configs()
        
        if self.simulate_checkbox.isChecked():
            for ch in self.channel_ui_configs:
                if ch['sensor_cb'].currentText() == "None":
                    ch['offset_input'].setText("0.000000")
            print("[INFO] Simulated offsets measured (0.0 V).")
            return

        try:
            rate = 1000
            samples = 1000 
            with nidaqmx.Task() as task:
                for ch in configs:
                    if ch['SensorType'] == "Type K":
                        task.ai_channels.add_ai_thrmcpl_chan(
                            ch['Terminal'],
                            thermocouple_type=ThermocoupleType.K,
                            cjc_source=CJCSource.BUILT_IN
                        )
                    else:
                        task.ai_channels.add_ai_voltage_chan(
                            ch['Terminal'],
                            terminal_config=ch['Config'],
                            min_val=ch['Range'][0],
                            max_val=ch['Range'][1]
                        )
                        
                task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=samples)
                data = task.read(number_of_samples_per_channel=samples, timeout=3.0)
                
                means = np.mean(data, axis=1)
                
                for i, ch_ui in enumerate(self.channel_ui_configs):
                    if configs[i]['SensorType'] == "None":
                        ch_ui['offset_input'].setText(f"{means[i]:.6f}")
                    
            print("[INFO] Offsets measured and updated in UI successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to measure offsets: {e}")

    def get_current_channel_configs(self):
        config_map = {
            "RSE": TerminalConfiguration.RSE,
            "NRSE": TerminalConfiguration.NRSE,
            "DIFF": TerminalConfiguration.DIFF,
            "PSEUDO_DIFF": TerminalConfiguration.PSEUDO_DIFF
        }
        range_map = {
            "-0.2 to 0.2": (-0.2, 0.2),
            "-2.5 to 2.5": (-2.5, 2.5),
            "-5 to 5": (-5.0, 5.0),
            "-10 to 10": (-10.0, 10.0)
        }
        
        daq_configs = []
        for ch in self.channel_ui_configs:
            sensor_type = ch['sensor_cb'].currentText()
            try:
                scale_val = float(ch['scale_input'].text())
            except ValueError:
                scale_val = 1.0 
            try:
                offset_val = float(ch['offset_input'].text())
            except ValueError:
                offset_val = 0.0
                
            if sensor_type != "None":
                scale_val = 1.0
                offset_val = 0.0

            daq_configs.append({
                'Name': ch['name'],
                'Terminal': ch['terminal_str'],
                'Config': config_map[ch['term_cb'].currentText()],
                'Range': range_map[ch['range_cb'].currentText()],
                'SensorType': sensor_type,
                'Scale': scale_val,
                'Unit': ch['unit_input'].text().strip(),
                'Offset': offset_val
            })
        return daq_configs

    # --- INDICATOR MANAGEMENT ---
    def add_indicator(self, default_signal="AI0", default_type="Current (100ms avg)"):
        widget = NumericalIndicatorWidget(self.available_signals, self.remove_indicator)
        if default_signal in self.available_signals:
            widget.signal_cb.setCurrentText(default_signal)
        widget.type_cb.setCurrentText(default_type)
        
        self.indicator_scroll_layout.insertWidget(len(self.indicator_widgets), widget)
        self.indicator_widgets.append(widget)

    def remove_indicator(self, widget):
        self.indicator_scroll_layout.removeWidget(widget)
        widget.deleteLater()
        self.indicator_widgets.remove(widget)

    def refresh_indicators(self):
        if not getattr(self, 'buffer_lock', threading.Lock()).locked():
            try:
                rate = float(self.read_rate_input.text())
            except ValueError:
                rate = 10000.0
                
            samples_05s = int(rate * 0.5)
            
            with self.buffer_lock:
                if self.sample_nr == 0 or self.full_resolution_buffer is None:
                    return
                n_extract = min(self.sample_nr, samples_05s)
                if n_extract == 0:
                    return
                recent_data_raw = self.full_resolution_buffer[:, -n_extract:].copy()
            
            current_configs = self.active_channel_configs
            scales = {cfg["Name"]: cfg["Scale"] for cfg in current_configs}
            units = {cfg["Name"]: cfg["Unit"] for cfg in current_configs}
            scales["DMM"] = 1.0
            units["DMM"] = "V"
            
            recent_scaled = {}
            for i in range(6):
                ch_name = f"AI{i}"
                recent_scaled[ch_name] = recent_data_raw[i, :] * scales.get(ch_name, 1.0)
            recent_scaled["DMM"] = recent_data_raw[6, :] * scales.get("DMM", 1.0)
            
            for ind in self.indicator_widgets:
                sig_name = ind.signal_cb.currentText()
                unit = units.get(sig_name, "")
                ind.calculate_and_update(recent_scaled.get(sig_name, []), rate, unit)


    # --- SUBPLOT MANAGEMENT ---
    def add_subplot(self, default_signals=None):
        idx = len(self.subplot_widgets)
        widget = SubplotConfigWidget(idx, self.available_signals, self.remove_subplot)
        if default_signals:
            for sig in default_signals:
                if sig in widget.checkboxes:
                    widget.checkboxes[sig].setChecked(True)
        
        self.plot_scroll_layout.insertWidget(len(self.subplot_widgets), widget)
        self.subplot_widgets.append(widget)
        self.rebuild_subplots()

    def remove_subplot(self, widget):
        self.plot_scroll_layout.removeWidget(widget)
        widget.deleteLater()
        self.subplot_widgets.remove(widget)
        for i, w in enumerate(self.subplot_widgets):
            w.update_index(i)
        self.rebuild_subplots()

    def rebuild_subplots(self):
        with self.plot_ui_lock:
            self.figure.clear()
            self.axs = []
            n = len(self.subplot_widgets)
            if n > 0:
                for i in range(n):
                    ax = self.figure.add_subplot(n, 1, i + 1)
                    self.axs.append(ax)
                self.figure.subplots_adjust(hspace=0.4)
            self.canvas.draw_idle()

    # --- DAQ LOGIC ---
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.folder_display.setText(f"Output Folder: {folder}")

    def generate_filename(self, base_name=""):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        default_name = f"{base_name.strip()}-{timestamp}.tdms" if base_name.strip() else f"{timestamp}.tdms"
        folder_path = Path(self.output_folder) if self.output_folder else Path.cwd()
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / default_name

    def generate_profile(self, write_rate):
        return [] 

    def start_write(self):
        if self.start_timestamp is None:
            self.start_timestamp = time.time()

        self.write_stop_flag.clear()
        self.exported = False

        if self.simulate_checkbox.isChecked():
            print("[INFO] Simulating AO write.")
            self.write_active_label.setStyleSheet("color: green; font-weight: bold;")
            self.shutdown_label.setText("Status: OK (Simulated)")
            self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")
            return

        write_rate = float(self.write_rate_input.text())
        current_profile = self.generate_profile(write_rate)
        voltages = [val / 200.0 for val in current_profile] 
        if voltages and voltages[-1] != 0.0:
            voltages.append(0.0)
        self.target_current_log = current_profile

        with self.write_task_lock:
            if self.write_task is not None:
                try: self.write_task.stop()
                except Exception: pass
                try: self.write_task.close()
                except Exception: pass
                self.write_task = None

            self.write_task = nidaqmx.Task()
            self.write_task.ao_channels.add_ao_voltage_chan(WRITE_CHANNEL)
            self.write_task.timing.cfg_samp_clk_timing(
                write_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(voltages)
            )

            try:
                number_of_samples_written = self.write_task.write(voltages, auto_start=True)
                print(f"Generating {number_of_samples_written} voltage samples.")
            except nidaqmx.errors.DaqError as e:
                print(f"[WARN] AO write failed: {e}")
                try: self.write_task.close()
                except Exception: pass
                self.write_task = None
                time.sleep(0.1)
                self.write_task = nidaqmx.Task()
                self.write_task.ao_channels.add_ao_voltage_chan(WRITE_CHANNEL)
                self.write_task.timing.cfg_samp_clk_timing(
                    write_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(voltages)
                )
                number_of_samples_written = self.write_task.write(voltages, auto_start=True)
                print(f"[INFO] Retry succeeded. Generating {number_of_samples_written} voltage samples.")

        self.write_active_label.setStyleSheet("color: green; font-weight: bold;")
        self.shutdown_label.setText("Status: OK")
        self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")

    def stop_write(self):
        self.write_stop_flag.set()
        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")

        if self.simulate_checkbox.isChecked():
            return

        with self.write_task_lock:
            try:
                if self.write_task is not None:
                    try: self.write_task.stop()
                    except Exception: pass
                    try: self.write_task.close()
                    except Exception: pass
                    self.write_task = None
                    print("[INFO] Write task closed.")
            except Exception as e:
                print(f"[ERROR] Failed to close write task: {e}")

        try:
            self.zero_ao_output()
        except Exception as e:
            print(f"[ERROR] Failed to zero AO0: {e}")

    def zero_ao_output(self):
        if self.simulate_checkbox.isChecked():
            self.last_output_voltage = 0.0
            return

        try:
            with nidaqmx.Task() as zero_task:
                zero_task.ao_channels.add_ao_voltage_chan(WRITE_CHANNEL)
                zero_task.write(0.0)
                time.sleep(0.01)
                zero_task.write(0.0)
            self.last_output_voltage = 0.0
            print("[INFO] AO0 successfully zeroed.")
        except Exception as e:
            print(f"[ERROR] Failed to zero AO0: {e}")

    def start_read(self):
        self.active_channel_configs = self.get_current_channel_configs()
        
        if self.DMMread_thread and self.DMMread_thread.is_alive(): return   
        if self.read_thread and self.read_thread.is_alive(): return     
        if self.plot_thread and self.plot_thread.is_alive(): return
        
        self.read_stop_flag.clear()
        if self.start_timestamp is None:
            self.start_timestamp = time.time()
            
        self.history_time.clear()
        for key in self.history_data:
            self.history_data[key].clear()

        self.buffer_lock = threading.Lock()
        self.dmmbuffer_lock = threading.Lock()
        
        self.DMMread_thread = threading.Thread(target=self.DMM_read, name="DMMread_thread")
        self.read_thread = threading.Thread(target=self.read_voltages, name="read_voltages")
        self.plot_thread = threading.Thread(target=self.plot_and_save, name="plot_and_save")
        self.DMMread_thread.start()
        time.sleep(2)
        self.read_thread.start()
        self.plot_thread.start()
        
        self.indicator_timer.start(200)

    def stop_read(self):
        self.read_stop_flag.set()
        self.indicator_timer.stop()

    def DMM_read(self):
        print("DMM starting")
        
        if self.simulate_checkbox.isChecked():
            print("[INFO] Simulating DMM reading.")
            while not self.read_stop_flag.is_set():
                self.dmmbuffer.append(float(np.random.uniform(-0.1, 0.1)))
                time.sleep(0.1)
            return

        inst = None
        try:
            inst = DMM6510readout.write_script_to_Keithley(self.Keithley_DMM_IP.text(), "0.05")
            while not self.read_stop_flag.is_set():
                data = DMM6510readout.read_data(inst)
                self.dmmbuffer.append(float(data))
        except Exception as e:
            pass
        finally:
            try:
                if inst is not None:
                    DMM6510readout.stop_instrument(inst)
            except Exception:
                pass

    def resize_dmmdata(self, length):
        data = np.asarray(self.dmmbuffer)
        n = len(data)
        self.dmmbuffer = self.dmmbuffer[-1:]
        if n == 0:
            return np.zeros(length)
        if n < length:
            block = length // n
            pattern = np.repeat(data, block)
            remaining = length - len(pattern)
            if remaining > 0:
                pattern = np.concatenate([pattern, np.repeat(data[-1], remaining)])
            return pattern
        else:
            indices = np.linspace(0, n, length+1, endpoint=True).astype(int)
            output = []
            for i in range(length):
                segment = data[indices[i]:indices[i+1]]
                output.append(segment.mean() if len(segment) > 0 else output[-1])
            return np.array(output)

    def read_voltages(self):
        current_configs = self.active_channel_configs
        ui_offsets = np.array([cfg['Offset'] for cfg in current_configs])[:, np.newaxis]

        try:
            self.sample_nr = 0
            read_rate = float(self.read_rate_input.text())
            samples_per_read = max(1, int(read_rate // 100))
            threshold = float(self.threshold_input.text())

            self.full_resolution_buffer = np.zeros((7, int(read_rate * 30)), dtype=np.float64)
            self.threshold_triggered = False

            # --- SIMULATION MODE LOOP ---
            if self.simulate_checkbox.isChecked():
                print("[INFO] Simulating AI reading.")
                t_wave = 0
                while not self.read_stop_flag.is_set():
                    t_start = time.time()
                    
                    time_arr = np.linspace(t_wave, t_wave + samples_per_read/read_rate, samples_per_read)
                    t_wave += samples_per_read/read_rate
                    
                    data = np.random.uniform(-0.1, 0.1, (6, samples_per_read))
                    data[0, :] += np.sin(2 * np.pi * 50 * time_arr) * 2.0  
                    data[1, :] += np.sin(2 * np.pi * 15 * time_arr) * 1.5 
                    
                    for i, cfg in enumerate(current_configs):
                        if cfg['SensorType'] == "Type K":
                            data[i, :] = np.random.uniform(24.5, 25.5, samples_per_read)
                            
                    data = data - ui_offsets 

                    n_samp = samples_per_read
                    
                    with self.buffer_lock:
                        self.full_resolution_buffer = np.roll(self.full_resolution_buffer, -samples_per_read, axis=1)
                        self.full_resolution_buffer[:6, -samples_per_read:] = data
                        self.full_resolution_buffer[6:, -samples_per_read:] = self.resize_dmmdata(samples_per_read)
                        self.sample_nr += n_samp
                    
                    elapsed = time.time() - t_start
                    sleep_time = (samples_per_read / read_rate) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                return
            # ---------------------------
            
            data = np.zeros((6, samples_per_read), dtype=np.float64)
            with nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
                for ch in current_configs:
                    if ch['SensorType'] == "Type K":
                        ai_task.ai_channels.add_ai_thrmcpl_chan(
                            ch['Terminal'],
                            thermocouple_type=ThermocoupleType.K,
                            cjc_source=CJCSource.BUILT_IN
                        )
                    else:
                        ai_task.ai_channels.add_ai_voltage_chan(
                            ch['Terminal'],
                            terminal_config=ch['Config'],
                            min_val=ch['Range'][0],
                            max_val=ch['Range'][1]
                        )

                ai_task.timing.cfg_samp_clk_timing(
                    rate=read_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=int(read_rate * 10)
                )
                terminal_name = get_terminal_name_with_dev_prefix(ai_task, "ai/SampleClock")

                do_task.do_channels.add_do_chan(DO_CHANNEL, line_grouping=LineGrouping.CHAN_PER_LINE)
                do_task.timing.cfg_samp_clk_timing(
                    1000.0, terminal_name, sample_mode=AcquisitionType.FINITE, samps_per_chan=2
                )
                do_task.write([[True, False], [False, False]])
                do_task.start()

                stream_reader = stream_readers.AnalogMultiChannelReader(ai_task.in_stream)

                t = time.time_ns()
                ai_task.start()
                time.sleep(0.1)
                do_task.stop()
                do_task.timing.cfg_samp_clk_timing(
                    1000.0, sample_mode=AcquisitionType.FINITE, samps_per_chan=2
                )
                do_task.write([[False, False], [True, True]])

                while not self.read_stop_flag.is_set():
                    try:
                        n_samp = stream_reader.read_many_sample(
                            data=data,
                            number_of_samples_per_channel=samples_per_read,
                            timeout=0.1
                        )
                        
                        data = data - ui_offsets

                    except Exception as e:
                        print(f"[ERROR] DAQ read failed: {e}")
                        continue

                    median_val = np.median(data[2, :])
                    if (abs(median_val) > threshold) and (not self.write_stop_flag.is_set()):
                        try:
                            do_task.start()
                        except Exception:
                            pass

                        with self.write_task_lock:
                            wt = self.write_task
                            if wt is not None:
                                try: wt.stop()
                                except Exception: pass
                                try: wt.close()
                                except Exception: pass
                                self.write_task = None

                        self.write_stop_flag.set()
                        print(f"[WARNING] Threshold exceeded on |AI2|: {median_val:.6f} > {threshold}")
                        self.shutdown_label.setText("Status: SHUTDOWN")
                        self.shutdown_label.setStyleSheet("color: red; font-weight: bold;")

                        try:
                            self.zero_ao_output()
                        except Exception as e:
                            print(f"[ERROR] Zero AO after shutdown failed: {e}")

                        time.sleep(0.001)
                        try:
                            do_task.stop()
                        except Exception:
                            pass
                        self.threshold_triggered = True

                    with self.buffer_lock:
                        self.full_resolution_buffer = np.roll(self.full_resolution_buffer, -samples_per_read, axis=1)
                        self.full_resolution_buffer[:6, -samples_per_read:] = data
                        self.full_resolution_buffer[6:, -samples_per_read:] = self.resize_dmmdata(samples_per_read)
                        self.sample_nr += n_samp

        except Exception as e:
            print(f"[ERROR] read_voltages crashed: {e}")
        finally:
            self.read_stop_flag.set()


    def plot_and_save(self):
        last_plot_sample = 0
        average_samples = max(1, int(self.average_samples_input.text()))
        filename = self.generate_filename(self.averaged_filename_input.text())
        
        current_configs = self.active_channel_configs
        scales = {cfg["Name"]: cfg["Scale"] for cfg in current_configs}
        scales["DMM"] = 1.0 

        while not self.read_stop_flag.is_set():
            if self.read_thread and not self.read_thread.is_alive():
                self.read_stop_flag.set()
                break

            if (self.sample_nr - last_plot_sample > average_samples) and (not getattr(self, 'buffer_lock', threading.Lock()).locked()):
                data_to_average = self.full_resolution_buffer[:, -(self.sample_nr - last_plot_sample):].copy()

                n_of_points = data_to_average.shape[1] // average_samples
                for i in range(n_of_points):
                    averaged_data_tmp = np.mean(
                        data_to_average[:, i * average_samples:(i + 1) * average_samples], axis=1, keepdims=True
                    )

                    timestamp = (last_plot_sample + average_samples / 2) / float(self.read_rate_input.text())
                    last_plot_sample += average_samples

                    self.history_time.append(timestamp)
                    self.history_data["AI0"].append(averaged_data_tmp[0, 0] * scales.get("AI0", 1.0))
                    self.history_data["AI1"].append(averaged_data_tmp[1, 0] * scales.get("AI1", 1.0))
                    self.history_data["AI2"].append(averaged_data_tmp[2, 0] * scales.get("AI2", 1.0))
                    self.history_data["AI3"].append(averaged_data_tmp[3, 0] * scales.get("AI3", 1.0))
                    self.history_data["AI4"].append(averaged_data_tmp[4, 0] * scales.get("AI4", 1.0))
                    self.history_data["AI5"].append(averaged_data_tmp[5, 0] * scales.get("AI5", 1.0))
                    self.history_data["DMM"].append(averaged_data_tmp[6, 0] * scales.get("DMM", 1.0))

                self.update_plot()

            if self.threshold_triggered and not self.exported:
                time.sleep(1)
                self.export_high_res_data()
                self.exported = True

            time.sleep(0.1)

        try:
            with TdmsWriter(str(filename)) as writer:
                writer.write_segment([ChannelObject("Group", "Time [s]", self.history_time)])
                
                units = {cfg["Name"]: cfg["Unit"] for cfg in current_configs}
                for ch_name in ["AI0", "AI1", "AI2", "AI3", "AI4", "AI5"]:
                    unit = units.get(ch_name, "V")
                    writer.write_segment([ChannelObject("Group", f"{ch_name} [{unit}]", self.history_data[ch_name])])
                
                writer.write_segment([ChannelObject("Group", "DMM [V]", self.history_data["DMM"])])
                
                if hasattr(self, "target_current_log"):
                    writer.write_segment([ChannelObject("Group", "Target_Current", np.array(self.target_current_log))])
                if self.threshold_triggered:
                    writer.write_segment([ChannelObject("Group", "Threshold_Triggered", np.array([1]))])
            print(f"[INFO] Averaged TDMS written: {filename}")
        except Exception as e:
            print(f"[ERROR] Final TDMS write failed: {e}")

    def reset_all(self):
        print("[INFO] Resetting state...")
        self.write_stop_flag.set()
        self.read_stop_flag.set()
        self.indicator_timer.stop()

        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=3)
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=3)

        with self.write_task_lock:
            wt = self.write_task
            if wt is not None:
                try: wt.stop()
                except Exception: pass
                try: wt.close()
                except Exception: pass
                self.write_task = None

        self.zero_ao_output()

        self.history_time.clear()
        for key in self.history_data:
            self.history_data[key].clear()

        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")
        self.shutdown_label.setText("Status: OK")
        self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")
        self.start_timestamp = None

        for ind in self.indicator_widgets:
            ind.value_label.setText("0.00")

        self.rebuild_subplots()

    def exit_application(self):
        print("[INFO] Exiting application. Stopping tasks and zeroing AO0...")
        self.write_stop_flag.set()
        self.read_stop_flag.set()
        self.indicator_timer.stop()
        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")

        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=3)
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=3)

        with self.write_task_lock:
            wt = self.write_task
            if wt is not None:
                try: wt.stop()
                except Exception: pass
                try: wt.close()
                except Exception: pass
                self.write_task = None

        self.zero_ao_output()
        self.start_timestamp = None
        QApplication.quit()

    def update_plot(self):
        try:
            window_s = float(self.plot_window_input.text())
        except ValueError:
            window_s = 10.0

        try:
            rate = float(self.read_rate_input.text())
            avg_samps = max(1, int(self.average_samples_input.text()))
            eff_plot_rate = rate / avg_samps
        except ValueError:
            eff_plot_rate = 100.0 

        num_points = int(window_s * eff_plot_rate) if window_s > 0 else 0

        current_configs = self.active_channel_configs
        units = {cfg["Name"]: cfg["Unit"] for cfg in current_configs}
        units["DMM"] = "V" 
        
        with self.plot_ui_lock:
            for i, ax in enumerate(self.axs):
                ax.clear()
                
                selected_signals = self.subplot_widgets[i].get_selected_signals()
                plot_units = set() 
                
                for sig in selected_signals:
                    if len(self.history_time) == len(self.history_data[sig]) and len(self.history_time) > 1:
                        if sig == "DMM" and all(v == 0 for v in self.history_data[sig]):
                            continue
                        
                        unit = units.get(sig, "V")
                        plot_units.add(unit)

                        # Fast List Slicing logic
                        if num_points > 0 and len(self.history_time) > num_points:
                            t_plot = self.history_time[-num_points:]
                            y_plot = self.history_data[sig][-num_points:]
                        else:
                            t_plot = self.history_time
                            y_plot = self.history_data[sig]

                        ax.plot(t_plot, y_plot, label=f"{sig} [{unit}]")
                
                if plot_units:
                    unit_str = ", ".join(sorted(list(plot_units)))
                    ax.set_ylabel(f"Value [{unit_str}]")
                else:
                    ax.set_ylabel("Value")
                
                if selected_signals:
                    ax.legend(loc='upper right')
                ax.set_xlim(left=0)
                ax.grid('on')
                
                if i == len(self.axs) - 1:
                    ax.set_xlabel("Time (s)")

            self.canvas.draw()

    def export_high_res_data(self):
        try:
            filename = self.generate_filename(self.averaged_filename_input.text()+"_manual_export")
            with TdmsWriter(str(filename)) as writer:
                self.buffer_lock.acquire()
                buffer_tmp = self.full_resolution_buffer.copy()
                self.buffer_lock.release()
                for i, ch in enumerate(buffer_tmp):
                    j = i
                    if i > 2:
                        j += 1
                    writer.write_segment([ChannelObject("Group", f"AI{j}", ch.copy())])
            print("[INFO] Manual full resolution data export complete.")
        except Exception as e:
            print(f"[ERROR] Failed manual export: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DAQControlApp()
    gui.show()
    sys.exit(app.exec_())
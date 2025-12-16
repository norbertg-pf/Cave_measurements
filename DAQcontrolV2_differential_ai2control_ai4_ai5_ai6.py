import sys
import threading
import nidaqmx.system
import numpy as np
import nidaqmx
from nidaqmx import stream_readers
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, LineGrouping, LoggingMode, LoggingOperation, ProductCategory
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QGridLayout,
                             QCheckBox, QListWidget, QListWidgetItem, QFileDialog)
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from nptdms import TdmsWriter, ChannelObject
from datetime import datetime
import collections
import time
from pathlib import Path
import DMM6510readout
import thermocouple_k
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


WRITE_CHANNEL = "Dev2/ao0"
DO_CHANNEL = "Dev2/port0/line0:1"
PROT_CHANNEL = "Dev2/ai1"
read_channels_factory = {
    'Terminals': ["Dev2/ai0", "Dev2/ai1", "Dev2/ai2", "Dev2/ai4", "Dev2/ai5", "Dev2/ai6"],
    'Ranges': [(-10,10), (-10,10), (-0.2,0.2), (-0.2,0.2), (-0.2,0.2), (-5,5)],
    'Config': [TerminalConfiguration.RSE, TerminalConfiguration.DIFF, TerminalConfiguration.DIFF, TerminalConfiguration.DIFF, TerminalConfiguration.DIFF, TerminalConfiguration.DIFF],
}
READ_CHANNELS = [{'Terminal': read_channels_factory['Terminals'][i],
                  'Range': read_channels_factory['Ranges'][i],
                  'Config': read_channels_factory['Config'][i]}
                 for i in range(len(read_channels_factory['Terminals']))]


class DAQControlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAQ Control GUI")

        self.full_resolution_buffer = None
        self.dmm_buffer = None
        self.dmmbuffer = []
        self.time_buffer = None
        self.write_thread = None
        self.DMMread_thread = None
        self.read_thread = None
        self.plot_thread = None
        self.write_stop_flag = threading.Event()
        self.read_stop_flag = threading.Event()
        self.last_output_voltage = 0.0
        self.output_folder = ""
        self.write_task = None

        # NEW: lock to guard access to self.write_task
        self.write_task_lock = threading.Lock()

        self.sim_latest_voltage = 0.0
        self.start_timestamp = None
        self.threshold_triggered = False
        self.exported = False
        self.sample_nr = 0

        self.ai_offsets = [0.0] * len(READ_CHANNELS)
        self.offset_calibrated = False
        self.profile_data = []

        self.plot_buffers = [[], [], [], [], [], [], []]
        self.time_buffers = {
            "write": [],
            "ai0ai1": [],
            "ai2": []
        }
        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        center_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        self.export_button = QPushButton("Export Last 30s Data")
        self.export_button.clicked.connect(self.export_high_res_data)
        right_panel.addWidget(self.export_button)

        controls_layout = QGridLayout()
        self.write_rate_input = QLineEdit("1000")
        self.read_rate_input = QLineEdit("10000")
        self.average_samples_input = QLineEdit("100")
        self.threshold_input = QLineEdit("0.0002")
        self.averaged_filename_input = QLineEdit("test")
        self.simulate_checkbox = QCheckBox("Simulate Mode")
        self.thermocouple_ai2_checkbox = QCheckBox("Thermocouple on ai2")
        self.thermocouple_ai4_checkbox = QCheckBox("Thermocouple on ai4")
        self.write_active_label = QLabel("Write Active")
        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")
        self.shutdown_label = QLabel("Status: OK")
        self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")
        self.calibrate_btn = QPushButton("Calibrate Offsets")
        self.calibrate_btn.clicked.connect(self.calibrate_offsets)

        choose_folder_btn = QPushButton("Choose Output Folder")
        choose_folder_btn.clicked.connect(self.select_output_folder)
        self.folder_display = QLabel()
        controls_layout.addWidget(QLabel("Write Rate (Hz):"), 0, 0)
        controls_layout.addWidget(self.write_rate_input, 0, 1)
        controls_layout.addWidget(QLabel("Read Rate (Hz):"), 1, 0)
        controls_layout.addWidget(self.read_rate_input, 1, 1)
        controls_layout.addWidget(QLabel("Samples to Average Over:"), 2, 0)
        controls_layout.addWidget(self.average_samples_input, 2, 1)
        controls_layout.addWidget(QLabel("Voltage Threshold (V):"), 3, 0)
        controls_layout.addWidget(self.threshold_input, 3, 1)
        controls_layout.addWidget(QLabel("Averaged Data Filename:"), 4, 0)
        controls_layout.addWidget(self.averaged_filename_input, 4, 1)
        controls_layout.addWidget(choose_folder_btn, 5, 0, 1, 2)
        controls_layout.addWidget(QLabel("Keithley DMM IP"), 6, 0)
        self.Keithley_DMM_IP = QLineEdit("169.254.169.37")
        controls_layout.addWidget(self.Keithley_DMM_IP, 6, 1)
        controls_layout.addWidget(self.thermocouple_ai2_checkbox, 7, 0)
        controls_layout.addWidget(self.thermocouple_ai4_checkbox, 7, 1)
        controls_layout.addWidget(self.folder_display, 8, 0, 1, 2)
        controls_layout.addWidget(self.write_active_label, 9, 0, 1, 2)
        controls_layout.addWidget(self.simulate_checkbox, 10, 0, 1, 2)
        controls_layout.addWidget(self.shutdown_label, 11, 0, 1, 2)
        controls_layout.addWidget(self.calibrate_btn, 12, 0, 1, 2)
        
        # Set default folder here
        self.output_folder = r"C:\Users\PF-test-stand\Documents\Development\Measurements\Heater_measurements\data"
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.folder_display.setText(f"Output Folder: {self.output_folder if len(self.output_folder)<40 else self.output_folder[-40:]}")

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

        self.figure, self.axs = plt.subplots(5, 1, figsize=(10, 10), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)

        self.profile_list = QListWidget()
        self.profile_entries = []
        self.current_input = QLineEdit("420")
        self.ramp_rate_input = QLineEdit("20")
        self.hold_time_input = QLineEdit("0")

        add_step_btn = QPushButton("Add Step")
        add_step_btn.clicked.connect(self.add_profile_step)
        build_plot_btn = QPushButton("Plot Current Profile")
        build_plot_btn.clicked.connect(self.plot_current_profile)
        clear_profile_btn = QPushButton("Clear Current Profile")
        clear_profile_btn.clicked.connect(self.clear_profile)

        left_panel.addWidget(QLabel("Current Profile Builder"))
        left_panel.addWidget(QLabel("Target Current (A):"))
        left_panel.addWidget(self.current_input)
        left_panel.addWidget(QLabel("Ramp Rate (A/s):"))
        left_panel.addWidget(self.ramp_rate_input)
        left_panel.addWidget(QLabel("Hold Time (s):"))
        left_panel.addWidget(self.hold_time_input)
        left_panel.addWidget(add_step_btn)
        left_panel.addWidget(build_plot_btn)
        left_panel.addWidget(clear_profile_btn)
        left_panel.addWidget(self.profile_list)
        right_panel.addLayout(controls_layout)
        center_panel.addWidget(self.canvas)
        center_panel.addLayout(btn_layout)

        main_layout.addLayout(left_panel)
        main_layout.addLayout(center_panel)
        main_layout.addLayout(right_panel)
        self.setLayout(main_layout)

        self.axs[0].set_title("Planned Current Profile")
        self.axs[0].set_ylabel("Current (A)")
        self.axs[0].set_xlabel("Time (s)")

        nidaqmx.system.Device('Dev2').reset_device()

        do_init = nidaqmx.Task()
        do_init.do_channels.add_do_chan(DO_CHANNEL, line_grouping=LineGrouping.CHAN_PER_LINE)
        do_init.write([False, False])
        do_init.close()

    def add_profile_step(self):
        try:
            current = float(self.current_input.text())
            ramp = float(self.ramp_rate_input.text())
            hold = float(self.hold_time_input.text())
            self.profile_entries.append((current, ramp, hold))
            self.profile_list.addItem(QListWidgetItem(f"To {current} A @ {ramp} A/s, hold {hold}s"))
        except ValueError as e:
            print(f"[ERROR] Invalid profile step input: {e}")

    def generate_profile(self, rate):
        profile = []
        last_current = 0.0
        for (target, ramp, hold) in self.profile_entries:
            step = 1.0 / rate
            ramp_time = abs(target - last_current) / ramp if ramp != 0 else 0
            ramp_steps = int(ramp_time * rate)
            hold_steps = int(hold * rate)
            if ramp_steps > 0:
                profile += list(np.linspace(last_current, target, ramp_steps))
            profile += [target] * hold_steps
            last_current = target
        return profile

    def plot_current_profile(self):
        try:
            write_rate = float(self.write_rate_input.text())
            profile = self.generate_profile(write_rate)
            time_axis = np.linspace(0, len(profile) / write_rate, len(profile))
            self.axs[0].clear()
            self.axs[0].plot(time_axis, profile)
            self.axs[0].set_title("Planned Current Profile")
            self.axs[0].set_ylabel("Current (A)")
            self.axs[0].set_xlabel("Time (s)")
            self.axs[0].set_xlim(left=0)
            self.canvas.draw()
        except Exception as e:
            print(f"[ERROR] Failed to plot current profile: {e}")

    def clear_profile(self):
        self.profile_entries.clear()
        self.profile_list.clear()
        self.axs[0].clear()
        self.axs[0].set_title("Planned Current Profile")
        self.axs[0].set_ylabel("Current (A)")
        self.axs[0].set_xlabel("Time (s)")
        self.axs[0].set_xlim(left=0)
        self.canvas.draw()

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

    def calibrate_offsets(self):
        try:
            samples = 20000
            with nidaqmx.Task() as task:
                for ch in READ_CHANNELS:
                    task.ai_channels.add_ai_voltage_chan(
                        ch['Terminal'],
                        terminal_config=ch['Config'],
                        min_val=ch['Range'][0],
                        max_val=ch['Range'][1]
                    )
                task.timing.cfg_samp_clk_timing(rate=10000, sample_mode=AcquisitionType.FINITE, samps_per_chan=samples)
                data = task.read(number_of_samples_per_channel=samples, timeout=5.0)
                self.ai_offsets = np.array([np.mean(ch_data) for ch_data in data])[:, np.newaxis]
                self.offset_calibrated = True
                print(f"[INFO] AI Offsets calibrated: {self.ai_offsets.ravel().tolist()}")
        except Exception as e:
            print(f"[ERROR] Offset calibration failed: {e}")

    def start_write(self):
        # Ensure start time
        if self.start_timestamp is None:
            self.start_timestamp = time.time()

        self.write_stop_flag.clear()
        self.exported = False

        write_rate = float(self.write_rate_input.text())
        current_profile = self.generate_profile(write_rate)
        voltages = [val / 200.0 for val in current_profile]
        if voltages and voltages[-1] != 0.0:
            voltages.append(0.0)
        self.target_current_log = current_profile

        # Create fresh AO task
        with self.write_task_lock:
            # Clean any stale task
            if self.write_task is not None:
                try:
                    self.write_task.stop()
                except Exception:
                    pass
                try:
                    self.write_task.close()
                except Exception:
                    pass
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
                try:
                    self.write_task.close()
                except Exception:
                    pass
                self.write_task = None
                time.sleep(0.1)
                # Retry once with a brand-new task
                self.write_task = nidaqmx.Task()
                self.write_task.ao_channels.add_ao_voltage_chan(WRITE_CHANNEL)
                self.write_task.timing.cfg_samp_clk_timing(
                    write_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(voltages)
                )
                number_of_samples_written = self.write_task.write(voltages, auto_start=True)
                print(f"[INFO] Retry succeeded. Generating {number_of_samples_written} voltage samples.")

        # Indicator
        self.write_active_label.setStyleSheet("color: green; font-weight: bold;")
        self.shutdown_label.setText("Status: OK")
        self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")

    def stop_write(self):
        self.write_stop_flag.set()
        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")

        with self.write_task_lock:
            try:
                if self.write_task is not None:
                    try:
                        self.write_task.stop()
                    except Exception:
                        pass
                    try:
                        self.write_task.close()
                    except Exception:
                        pass
                    self.write_task = None
                    print("[INFO] Write task closed.")
            except Exception as e:
                print(f"[ERROR] Failed to close write task: {e}")

        # Always attempt to zero after releasing AO
        try:
            self.zero_ao_output()
        except Exception as e:
            print(f"[ERROR] Failed to zero AO0: {e}")

    def zero_ao_output(self):
        """Safely force AO to 0 V using a fresh, short-lived task."""
        try:
            with nidaqmx.Task() as zero_task:
                zero_task.ao_channels.add_ao_voltage_chan(WRITE_CHANNEL)
                zero_task.write(0.0)
                time.sleep(0.01)  # let hardware settle
                zero_task.write(0.0)
            self.last_output_voltage = 0.0
            print("[INFO] AO0 successfully zeroed.")
        except Exception as e:
            print(f"[ERROR] Failed to zero AO0: {e}")

    def start_read(self):
        self.calibrate_offsets()
        
        if self.DMMread_thread and self.DMMread_thread.is_alive():
            return   
        if self.read_thread and self.read_thread.is_alive():
            return     
        if self.plot_thread and self.plot_thread.is_alive():
            return
        self.read_stop_flag.clear()
        if self.start_timestamp is None:
            self.start_timestamp = time.time()
        self.time_buffers["ai0ai1"].clear()
        self.time_buffers["ai2"].clear()
        self.plot_buffers[2].clear()
        self.plot_buffers[3].clear()
        self.plot_buffers[4].clear()
        self.buffer_lock = threading.Lock()
        self.dmmbuffer_lock = threading.Lock()
        self.plot_lock = threading.Lock()
        self.DMMread_thread = threading.Thread(target=self.DMM_read, name="DMMread_thread")
        self.read_thread = threading.Thread(target=self.read_voltages, name="read_voltages")
        self.plot_thread = threading.Thread(target=self.plot_and_save, name="plot_and_save")
        self.DMMread_thread.start()
        time.sleep(2)
        self.read_thread.start()
        self.plot_thread.start()

    def stop_read(self):
        self.read_stop_flag.set()

    def DMM_read(self):
        print("DMM starting")
        try:
            #self.dmmbuffer.append(float(0))
            inst = DMM6510readout.write_script_to_Keithley(self.Keithley_DMM_IP.text(), "0.05")
            while not self.read_stop_flag.is_set():
                data = DMM6510readout.read_data(inst)
                self.dmmbuffer.append(float(data))

        except Exception as e:
            #print(f"[ERROR] read_voltages crashed: {e}")
            #traceback.print_exc()
            pass

        finally:
            # Always let the plot/save thread finish and write whatever we have
            # self.read_stop_flag.set()
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
        # Return zeros, NaNs or last known reading — your choice
        # Here I return zeros with correct length
            return np.zeros(length)
        # Case 1: expanding (n < length)
        #print(n, data)
        if n < length:
            block = length // n
            pattern = np.repeat(data, block)
            remaining = length - len(pattern)
            if remaining > 0:
                pattern = np.concatenate([pattern, np.repeat(data[-1], remaining)])
            #print(pattern)
            return pattern

        # Case 2: shrinking (n > length)
        else:
            # Create 100 bins with equal spacing across the original data
            indices = np.linspace(0, n, length+1, endpoint=True).astype(int)
            output = []
            for i in range(length):
                segment = data[indices[i]:indices[i+1]]
                output.append(segment.mean() if len(segment) > 0 else output[-1])
            #print(np.array(output))
            return np.array(output)

    def read_voltages(self):
        if self.thermocouple_ai2_checkbox.isChecked():
            thermocouple_ai2 = thermocouple_k.TypeKThermocouple(cjc_temp_C=23.0)
        if self.thermocouple_ai4_checkbox.isChecked():
            thermocouple_ai4 = thermocouple_k.TypeKThermocouple(cjc_temp_C=23.0)
        
        try:
            self.sample_nr = 0
            read_rate = float(self.read_rate_input.text())
            samples_per_read = max(1, int(read_rate // 100))
            threshold = float(self.threshold_input.text())

            data = np.zeros((6, samples_per_read), dtype=np.float64)
            self.full_resolution_buffer = np.zeros((7, int(read_rate * 30)), dtype=np.float64)
            self.threshold_triggered = False

            with nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
                # AI channels
                for ch in READ_CHANNELS:
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

                # DO protection lines, synced to AI sample clock at init
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
                        if self.offset_calibrated:
                            data = data - self.ai_offsets

                        if self.thermocouple_ai2_checkbox.isChecked():
                            data[2,:] = thermocouple_ai2.mV_to_tC(data[2,:]*1000 )
                        if self.thermocouple_ai4_checkbox.isChecked():
                            data[4,:] = thermocouple_ai4.mV_to_tC(data[4,:]*1000)
                    except Exception as e:
                        print(f"[ERROR] DAQ read failed: {e}")
                        continue

                    median_val = np.median(data[2, :])
                    if (abs(median_val) > threshold) and (not self.write_stop_flag.is_set()):
                        # Trip protection lines
                        try:
                            do_task.start()
                        except Exception:
                            pass

                        # Safely stop/close AO if it exists
                        with self.write_task_lock:
                            wt = self.write_task
                            if wt is not None:
                                try:
                                    wt.stop()
                                except Exception:
                                    pass
                                try:
                                    wt.close()
                                except Exception:
                                    pass
                                self.write_task = None

                        self.write_stop_flag.set()
                        print(f"[WARNING] Threshold exceeded on |AI2|: {median_val:.6f} V > {threshold} V")
                        try:
                            # print(f"Current: {np.max(data[1, :]/8.5*4250)} A")
                            print(f"Current: {np.max(data[1, :]*500)} A")
                        except Exception:
                            pass

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

                    # Update rolling buffer
                    with self.buffer_lock:
                        self.full_resolution_buffer = np.roll(self.full_resolution_buffer, -samples_per_read, axis=1)
                        self.full_resolution_buffer[:6, -samples_per_read:] = data
                        self.full_resolution_buffer[6:, -samples_per_read:] = self.resize_dmmdata(samples_per_read)
                        self.sample_nr += n_samp

                dt = time.time_ns() - t
                print(f"samples: {self.sample_nr}, dt: {dt/1000000} ms")

        except Exception as e:
            print(f"[ERROR] read_voltages crashed: {e}")
        finally:
            # Always let the plot/save thread finish and write whatever we have
            self.read_stop_flag.set()


    def plot_and_save(self):
        all_data = np.zeros([7, 1])

        last_plot_sample = 0
        average_samples = max(1, int(self.average_samples_input.text()))
        filename = self.generate_filename(self.averaged_filename_input.text())

        while not self.read_stop_flag.is_set():
            # If read thread died unexpectedly, exit and save
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

                    all_data = np.hstack([all_data, averaged_data_tmp])
                    timestamp = (last_plot_sample + average_samples / 2) / float(self.read_rate_input.text())
                    last_plot_sample += average_samples

                    ai0_val = averaged_data_tmp[0, 0] * 200  # Current = volt*200
                    # diff1 = averaged_data_tmp[1, 0] / 8.5 * 4250   # scaled current
                    diff1 = averaged_data_tmp[1, 0] * 500   # scaled current
                    diff2 = averaged_data_tmp[2, 0]
                    diff3 = averaged_data_tmp[3, 0]
                    diff4 = averaged_data_tmp[4, 0]
                    diff5 = averaged_data_tmp[5, 0]
                    diff6 = averaged_data_tmp[6, 0]

                    self.plot_buffers[2].append((ai0_val, diff1))
                    self.plot_buffers[3].append((diff2, diff3, diff4, diff5))
                    self.plot_buffers[4].append((diff6))
                    self.time_buffers["ai0ai1"].append(timestamp)
                    self.time_buffers["ai2"].append(timestamp)

                self.update_plot()

            if self.threshold_triggered and not self.exported:
                time.sleep(1)
                self.export_high_res_data()
                self.exported = True

            time.sleep(0.1)

        # Final write on exit
        try:
            with TdmsWriter(str(filename)) as writer:
                writer.write_segment([ChannelObject("Group", "Time [s]", self.time_buffers["ai2"][:])])
                for i, ch_data in enumerate(all_data):
                    j = i  # label ai channels correctly
                    if i > 2:
                        j += 1
                    writer.write_segment([ChannelObject("Group", f"AI{j}", ch_data)])
                    if i == 0:
                        writer.write_segment([ChannelObject("Group", "Target current [A]", ch_data * 200)])
                    if i == 1:
                        # writer.write_segment([ChannelObject("Group", "Measured current [A]", ch_data / 8.5 * 4250)])
                        writer.write_segment([ChannelObject("Group", "Measured current [A]", ch_data * 500)])
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

        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=3)
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=3)

        # Safely close AO
        with self.write_task_lock:
            wt = self.write_task
            if wt is not None:
                try:
                    wt.stop()
                except Exception:
                    pass
                try:
                    wt.close()
                except Exception:
                    pass
                self.write_task = None

        # Best-effort zero
        self.zero_ao_output()

        # Clear buffers/state
        self.plot_buffers = [[], [], [], [], [], [], []]
        for key in self.time_buffers:
            self.time_buffers[key].clear()

        self.profile_entries.clear()
        self.profile_list.clear()

        self.axs[0].clear()
        self.axs[0].set_title("Planned Current Profile")
        self.axs[0].set_ylabel("Current (A)")
        self.axs[0].set_xlabel("Time (s)")

        for i in range(1, 4):
            self.axs[i].clear()
            self.axs[i].set_ylabel("")

        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")
        self.shutdown_label.setText("Status: OK")
        self.shutdown_label.setStyleSheet("color: green; font-weight: bold;")
        self.start_timestamp = None

        self.canvas.draw()

    def exit_application(self):
        print("[INFO] Exiting application. Stopping tasks and zeroing AO0...")
        self.write_stop_flag.set()
        self.read_stop_flag.set()
        self.write_active_label.setStyleSheet("color: grey; font-weight: bold;")

        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=3)
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=3)

        with self.write_task_lock:
            wt = self.write_task
            if wt is not None:
                try:
                    wt.stop()
                except Exception:
                    pass
                try:
                    wt.close()
                except Exception:
                    pass
                self.write_task = None

        self.zero_ao_output()
        self.start_timestamp = None
        QApplication.quit()

    def update_plot(self):
        def safe_plot(ax, tdata, ydata, label=None):
            if len(tdata) == len(ydata) and len(tdata) > 1:
                ax.plot(tdata, ydata, label=label)

        # AI0 (target) and DCCT derived current
        self.axs[2].clear()
        ai0 = [v[0] for v in self.plot_buffers[2]]
        ai1_ai9 = [v[1] for v in self.plot_buffers[2]]
        safe_plot(self.axs[2], self.time_buffers["ai0ai1"], ai0, label="DO set (A)")
        safe_plot(self.axs[2], self.time_buffers["ai0ai1"], ai1_ai9, label="DCCT (A)")
        self.axs[2].set_ylabel("Current (A)")
        self.axs[2].legend()
        self.axs[2].set_xlim(left=0)
        self.axs[2].grid('on')

        # Differential voltages: AI2, AI4, AI5, AI6
        self.axs[3].clear()
        ai2_vals = [v[0] for v in self.plot_buffers[3]]
        ai4_vals = [v[1] for v in self.plot_buffers[3]]
        ai5_vals = [v[2] for v in self.plot_buffers[3]]
        ai6_vals = [v[3] for v in self.plot_buffers[3]]
       
        safe_plot(self.axs[3], self.time_buffers["ai2"], ai2_vals, label="AI2 (V)")
        safe_plot(self.axs[3], self.time_buffers["ai2"], ai4_vals, label="AI4 (V)")
        safe_plot(self.axs[3], self.time_buffers["ai2"], ai5_vals, label="AI5 (V)")
        safe_plot(self.axs[3], self.time_buffers["ai2"], ai6_vals, label="AI6 (V)")
        self.axs[3].set_ylabel("Differential Voltages (V)")
        self.axs[3].legend()
        self.axs[3].set_xlim(left=0)
        self.axs[3].grid('on')

        self.axs[4].clear()
        dmm_vals = [v for v in self.plot_buffers[4]]
        if not all(v == 0 for v in dmm_vals):
            safe_plot(self.axs[4], self.time_buffers["ai2"], dmm_vals, label="dmm (V)")


        self.canvas.draw()

    def export_high_res_data(self):
        try:
            filename = self.generate_filename(self.averaged_filename_input.text()+"_manual_export")
            with TdmsWriter(str(filename)) as writer:
                self.buffer_lock.acquire()
                buffer_tmp = self.full_resolution_buffer.copy()
                self.buffer_lock.release()
                for i, ch in enumerate(buffer_tmp):
                    j = i  # label ai channels correctly
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

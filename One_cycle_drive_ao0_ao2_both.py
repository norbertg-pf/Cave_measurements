import sys
import time
from typing import List

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QDoubleSpinBox, QFormLayout
)

import nidaqmx

# Defaults
DEFAULT_VOLTAGE = 10.0   # V  (pulse target)
DEFAULT_DURATION = 0.2   # s
PARK_VOLTAGE   = -0.0   # V  (baseline before/after pulse)
DEVICE = "Dev1"          # change if needed
AO0 = f"{DEVICE}/ao0"
AO2 = f"{DEVICE}/ao2"


class PulseSignals(QObject):
    status = pyqtSignal(str)      # "HIGH" | "LOW" | "ERROR"
    done = pyqtSignal()           # emitted when thread completes


class PulseWorker(QThread):
    """
    A QThread that performs a single HIGH pulse on 1+ AO channels, then returns to PARK_VOLTAGE.
    """
    def __init__(self, channels: List[str], voltage: float, duration_s: float, parent=None):
        super().__init__(parent)
        self.channels = channels
        self.voltage = float(voltage)
        self.duration_s = float(duration_s)
        self._stop = False
        self.signals = PulseSignals()

    def stop(self):
        self._stop = True

    def _write_all(self, task, value: float):
        # If multiple channels, write a list matching channel count
        if len(self.channels) > 1:
            task.write([value] * len(self.channels))
        else:
            task.write(value)

    def run(self):
        try:
            with nidaqmx.Task() as task:
                # Add requested channels to a SINGLE task (so multi-AO can be simultaneous)
                for ch in self.channels:
                    # Make sure the device range supports ±10 V; adjust if your HW differs.
                    task.ao_channels.add_ao_voltage_chan(ch, min_val=-10.0, max_val=10.0)

                # Park at -10 V before starting
                self._write_all(task, PARK_VOLTAGE)
                self.signals.status.emit("LOW")  # "LOW" == "PARK (-10 V)" in UI

                if self._stop:
                    self._write_all(task, PARK_VOLTAGE)
                    self.signals.status.emit("LOW")
                    return

                # Pulse to +voltage
                self._write_all(task, self.voltage)
                self.signals.status.emit("HIGH")

                t0 = time.perf_counter()
                while not self._stop and (time.perf_counter() - t0) < self.duration_s:
                    # light wait; keeps thread responsive to .stop()
                    time.sleep(0.005)

                # Return to park
                self._write_all(task, PARK_VOLTAGE)
                self.signals.status.emit("LOW")

                # small settle
                time.sleep(0.05)

        except Exception as e:
            print(f"[ERROR] PulseWorker on {self.channels}: {e}")
            self.signals.status.emit("ERROR")
        finally:
            self.signals.done.emit()


class PulseColumn(QWidget):
    """
    One column of controls: voltage, duration, Start/Stop, and status badge.
    Backed by a PulseWorker; safe to reuse per run.
    """
    def __init__(self, title: str, channels: List[str], parent=None):
        super().__init__(parent)
        self.channels = channels
        self.worker: PulseWorker = None

        group = QGroupBox(title)
        outer = QVBoxLayout(group)

        # Inputs
        form = QFormLayout()
        self.voltage_spin = QDoubleSpinBox()
        self.voltage_spin.setDecimals(3)
        # You can allow negative pulses if you want, but for +10 V pulses keep [0, 10].
        self.voltage_spin.setRange(0.0, 10.0)
        self.voltage_spin.setSingleStep(0.1)
        self.voltage_spin.setValue(DEFAULT_VOLTAGE)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setDecimals(3)
        self.duration_spin.setRange(0.01, 30)
        self.duration_spin.setSingleStep(0.05)
        self.duration_spin.setValue(DEFAULT_DURATION)

        form.addRow("Pulse to (V):", self.voltage_spin)
        form.addRow("Duration (s):", self.duration_spin)
        outer.addLayout(form)

        # Status
        self.status_label = QLabel("Voltage: PARK (0 V)")
        self.status_label.setAlignment(Qt.AlignCenter)
        self._set_style("LOW")
        outer.addWidget(self.status_label)

        # Buttons
        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet("background-color: green; color: white; font-size: 16px; padding: 8px;")
        self.start_btn.clicked.connect(self.start_pulse)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: red; color: white; font-size: 16px; padding: 8px;")
        self.stop_btn.clicked.connect(self.stop_pulse)

        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        outer.addLayout(btns)

        lay = QVBoxLayout(self)
        lay.addWidget(group)

    # --- UI helpers ---
    def _set_style(self, state: str):
        if state == "HIGH":
            self.status_label.setText("Voltage: HIGH (pulse active)")
            self.status_label.setStyleSheet("background-color: green; color: white; font-size: 14px; padding: 8px;")
        elif state == "LOW":
            self.status_label.setText(f"Voltage: PARK ({PARK_VOLTAGE:.1f} V)")
            self.status_label.setStyleSheet("background-color: gray; color: black; font-size: 14px; padding: 8px;")
        elif state == "ERROR":
            self.status_label.setText("Error: Check DAQ")
            self.status_label.setStyleSheet("background-color: orange; color: black; font-size: 14px; padding: 8px;")

    def _bind_worker(self, worker: PulseWorker):
        worker.signals.status.connect(self._set_style)
        worker.signals.done.connect(self._on_done)

    # --- Actions ---
    def start_pulse(self):
        # avoid double-start
        if self.worker is not None and self.worker.isRunning():
            return

        voltage = self.voltage_spin.value()
        duration = self.duration_spin.value()

        self.worker = PulseWorker(self.channels, voltage, duration)
        self._bind_worker(self.worker)

        self.start_btn.setEnabled(False)
        self.worker.start()

    def stop_pulse(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()

    def _on_done(self):
        self.start_btn.setEnabled(True)

    def close(self):
        # ensure thread ends before closing
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        super().close()


class VoltageToggleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAQ Pulse Control")
        self.setFixedSize(720, 300)

        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        # Three columns: ao0, ao2, ao0+ao2
        self.col_ao0 = PulseColumn("Pulse: ao0", [AO0])
        self.col_ao2 = PulseColumn("Pulse: ao2", [AO2])
        self.col_both = PulseColumn("Pulse: ao0 + ao2", [AO0, AO2])

        grid.addWidget(self.col_ao0, 0, 0)
        grid.addWidget(self.col_ao2, 0, 1)
        grid.addWidget(self.col_both, 0, 2)

        # Footer hint
        hint = QLabel(
            "Tip: Edit pulse target (V) and duration (s), then press Start. "
            f"Outputs park at {PARK_VOLTAGE:.1f} V before/after pulses. Stop returns to park."
        )
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)

        root = QVBoxLayout(self)
        root.addLayout(grid)
        root.addWidget(hint)

    def closeEvent(self, event):
        # Gracefully stop any running workers
        for col in (self.col_ao0, self.col_ao2, self.col_both):
            col.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoltageToggleApp()
    window.show()
    sys.exit(app.exec_())


# import sys
# import time
# from typing import List

# from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
#     QPushButton, QLabel, QGroupBox, QDoubleSpinBox, QFormLayout
# )

# import nidaqmx

# # Defaults
# DEFAULT_VOLTAGE = 10.0   # V
# DEFAULT_DURATION = 0.2   # s
# DEVICE = "Dev2"          # change if needed
# AO0 = f"{DEVICE}/ao0"
# AO2 = f"{DEVICE}/ao2"


# class PulseSignals(QObject):
#     status = pyqtSignal(str)      # "HIGH" | "LOW" | "ERROR"
#     done = pyqtSignal()           # emitted when thread completes


# class PulseWorker(QThread):
#     """
#     A QThread that performs a single HIGH pulse on 1+ AO channels, then returns LOW.
#     """
#     def __init__(self, channels: List[str], voltage: float, duration_s: float, parent=None):
#         super().__init__(parent)
#         self.channels = channels
#         self.voltage = float(voltage)
#         self.duration_s = float(duration_s)
#         self._stop = False
#         self.signals = PulseSignals()

#     def stop(self):
#         self._stop = True

#     def _write_all(self, task, value: float):
#         # If multiple channels, write a list matching channel count
#         if len(self.channels) > 1:
#             task.write([value] * len(self.channels))
#         else:
#             task.write(value)

#     def run(self):
#         try:
#             with nidaqmx.Task() as task:
#                 # Add requested channels to a SINGLE task (so multi-AO can be simultaneous)
#                 for ch in self.channels:
#                     task.ao_channels.add_ao_voltage_chan(ch)

#                 # Safety: force LOW before starting
#                 self._write_all(task, 0.0)

#                 if self._stop:
#                     self._write_all(task, 0.0)
#                     self.signals.status.emit("LOW")
#                     return

#                 # HIGH
#                 self._write_all(task, self.voltage)
#                 self.signals.status.emit("HIGH")

#                 t0 = time.perf_counter()
#                 while not self._stop and (time.perf_counter() - t0) < self.duration_s:
#                     # light wait; keeps thread responsive to .stop()
#                     time.sleep(0.005)

#                 # LOW
#                 self._write_all(task, 0.0)
#                 self.signals.status.emit("LOW")

#                 # small settle
#                 time.sleep(0.05)

#         except Exception as e:
#             print(f"[ERROR] PulseWorker on {self.channels}: {e}")
#             self.signals.status.emit("ERROR")
#         finally:
#             self.signals.done.emit()


# class PulseColumn(QWidget):
#     """
#     One column of controls: voltage, duration, Start/Stop, and status badge.
#     Backed by a PulseWorker; safe to reuse per run.
#     """
#     def __init__(self, title: str, channels: List[str], parent=None):
#         super().__init__(parent)
#         self.channels = channels
#         self.worker: PulseWorker = None

#         group = QGroupBox(title)
#         outer = QVBoxLayout(group)

#         # Inputs
#         form = QFormLayout()
#         self.voltage_spin = QDoubleSpinBox()
#         self.voltage_spin.setDecimals(3)
#         self.voltage_spin.setRange(0.0, 10.0)         # typical NI AO range; adjust to your HW
#         self.voltage_spin.setSingleStep(0.1)
#         self.voltage_spin.setValue(DEFAULT_VOLTAGE)

#         self.duration_spin = QDoubleSpinBox()
#         self.duration_spin.setDecimals(3)
#         self.duration_spin.setRange(0.01, 1.5)       # reasonable bounds; adjust as needed
#         self.duration_spin.setSingleStep(0.05)
#         self.duration_spin.setValue(DEFAULT_DURATION)

#         form.addRow("Voltage (V):", self.voltage_spin)
#         form.addRow("Duration (s):", self.duration_spin)
#         outer.addLayout(form)

#         # Status
#         self.status_label = QLabel("Voltage: LOW")
#         self.status_label.setAlignment(Qt.AlignCenter)
#         self._set_style("LOW")
#         outer.addWidget(self.status_label)

#         # Buttons
#         btns = QHBoxLayout()
#         self.start_btn = QPushButton("Start")
#         self.start_btn.setStyleSheet("background-color: green; color: white; font-size: 16px; padding: 8px;")
#         self.start_btn.clicked.connect(self.start_pulse)

#         self.stop_btn = QPushButton("Stop")
#         self.stop_btn.setStyleSheet("background-color: red; color: white; font-size: 16px; padding: 8px;")
#         self.stop_btn.clicked.connect(self.stop_pulse)

#         btns.addWidget(self.start_btn)
#         btns.addWidget(self.stop_btn)
#         outer.addLayout(btns)

#         lay = QVBoxLayout(self)
#         lay.addWidget(group)

#     # --- UI helpers ---
#     def _set_style(self, state: str):
#         if state == "HIGH":
#             self.status_label.setText("Voltage: HIGH")
#             self.status_label.setStyleSheet("background-color: green; color: white; font-size: 14px; padding: 8px;")
#         elif state == "LOW":
#             self.status_label.setText("Voltage: LOW")
#             self.status_label.setStyleSheet("background-color: gray; color: black; font-size: 14px; padding: 8px;")
#         elif state == "ERROR":
#             self.status_label.setText("Error: Check DAQ")
#             self.status_label.setStyleSheet("background-color: orange; color: black; font-size: 14px; padding: 8px;")

#     def _bind_worker(self, worker: PulseWorker):
#         worker.signals.status.connect(self._set_style)
#         worker.signals.done.connect(self._on_done)

#     # --- Actions ---
#     def start_pulse(self):
#         # avoid double-start
#         if self.worker is not None and self.worker.isRunning():
#             return

#         voltage = self.voltage_spin.value()
#         duration = self.duration_spin.value()

#         self.worker = PulseWorker(self.channels, voltage, duration)
#         self._bind_worker(self.worker)

#         self.start_btn.setEnabled(False)
#         self.worker.start()

#     def stop_pulse(self):
#         if self.worker is not None and self.worker.isRunning():
#             self.worker.stop()

#     def _on_done(self):
#         self.start_btn.setEnabled(True)

#     def close(self):
#         # ensure thread ends before closing
#         if self.worker is not None and self.worker.isRunning():
#             self.worker.stop()
#             self.worker.wait(2000)
#         super().close()


# class VoltageToggleApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("DAQ Pulse Control")
#         self.setFixedSize(720, 300)

#         grid = QGridLayout()
#         grid.setColumnStretch(0, 1)
#         grid.setColumnStretch(1, 1)
#         grid.setColumnStretch(2, 1)

#         # Three columns: ao0, ao2, ao0+ao2
#         self.col_ao0 = PulseColumn("Pulse: ao0", [AO0])
#         self.col_ao2 = PulseColumn("Pulse: ao2", [AO2])
#         self.col_both = PulseColumn("Pulse: ao0 + ao2", [AO0, AO2])

#         grid.addWidget(self.col_ao0, 0, 0)
#         grid.addWidget(self.col_ao2, 0, 1)
#         grid.addWidget(self.col_both, 0, 2)

#         # Footer hint
#         hint = QLabel("Tip: Edit Voltage (V) and Duration (s), then press Start. Stop will drop outputs to 0 V.")
#         hint.setWordWrap(True)
#         hint.setAlignment(Qt.AlignCenter)

#         root = QVBoxLayout(self)
#         root.addLayout(grid)
#         root.addWidget(hint)

#     def closeEvent(self, event):
#         # Gracefully stop any running workers
#         for col in (self.col_ao0, self.col_ao2, self.col_both):
#             col.close()
#         event.accept()


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = VoltageToggleApp()
#     window.show()
#     sys.exit(app.exec_())

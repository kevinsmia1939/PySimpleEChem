#!/usr/bin/python3
import sys
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyqtgraph as pg
from galvani import BioLogic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QHBoxLayout, QTableView, QComboBox, QSlider, QLabel,
    QPushButton, QFileDialog, QMenu, QAction, QCheckBox
)
from PyQt5.QtCore import Qt
from superqt import QRangeSlider
from function_collection import (
    battery_xls2df, get_CV_init, cy_idx_state_range,
    read_cv_format, get_peak_CV, search_pattern, ir_compen_func,
    diffusion, reaction_rate, peak_2nd_deriv, find_alpha,
    min_max_peak, check_val, switch_val, RDE_kou_lev,
    linear_fit, data_poly_inter, open_battery_data,
    df_select_column, read_cv_versastat,
    smooth_current_lowess  # <--- Added smoothing import
)
from scipy.signal import savgol_filter

pg.setConfigOption('background', 'white')
pg.setConfigOption('antialias', True)


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.5f}"
            return str(value)
        return None

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Plotter")

        # Plot widgets
        self.plot_E_I = pg.PlotWidget()
        self.plot_E_I.setLabel('left', text='I')
        self.plot_E_I.setLabel('bottom', text='E')
        self.plot_E_I.getAxis('bottom').setTextPen('black')
        self.plot_E_I.getAxis('left').setTextPen('black')

        self.plot_EV_I = pg.PlotWidget()
        self.plot_EV_I.setLabel('left', text='E/I')
        self.plot_EV_I.setLabel('bottom', text='1/I')
        self.plot_EV_I.getAxis('bottom').setTextPen('black')
        self.plot_EV_I.getAxis('left').setTextPen('black')

        # Controls: Open button and slider
        control_layout = QVBoxLayout()
        open_button_layout = QHBoxLayout()

        self.open_button = QPushButton("Add/Open CV file", self)
        self.open_button.setMenu(self.create_open_menu())

        self.delete_button = QPushButton("Delete selected", self)
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_selected_file)

        self.lsvchoosecombo = QComboBox(self)
        self.lsvchoosecombo.setFixedSize(300, 35)
        self.lsvchoosecombo.setEditable(False)
        self.lsvchoosecombo.setEnabled(False)
        self.lsvchoosecombo.currentIndexChanged.connect(self.choose_lsv)

        open_button_layout.addWidget(self.open_button)
        open_button_layout.addWidget(self.delete_button)
        open_button_layout.addWidget(self.lsvchoosecombo)
        control_layout.addLayout(open_button_layout)

        # Smoothing UI
        smoothing_layout = QHBoxLayout()
        self.smoothing_label = QLabel("Smoothing:", self)
        self.smoothing_method = QComboBox(self)
        self.smoothing_method.addItems(["None", "LOWESS", "Savitzky-Golay"])
        self.smoothing_method.setCurrentText("LOWESS")
        self.smoothing_method.currentTextChanged.connect(self.on_smoothing_changed)

        self.lowess_edit = QtWidgets.QLineEdit("0.1", self)
        self.lowess_edit.setFixedWidth(80)
        self.lowess_edit.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 3))
        self.lowess_edit.editingFinished.connect(self.on_smoothing_changed)

        self.savgol_window_edit = QtWidgets.QLineEdit("5", self)
        self.savgol_window_edit.setFixedWidth(60)
        self.savgol_window_edit.setValidator(QtGui.QIntValidator(1, 999))
        self.savgol_window_edit.editingFinished.connect(self.on_smoothing_changed)

        self.savgol_poly_edit = QtWidgets.QLineEdit("2", self)
        self.savgol_poly_edit.setFixedWidth(60)
        self.savgol_poly_edit.setValidator(QtGui.QIntValidator(1, 10))
        self.savgol_poly_edit.editingFinished.connect(self.on_smoothing_changed)

        self.smoothing_enable_checkbox = QCheckBox("Enable")
        self.smoothing_enable_checkbox.setChecked(False)
        self.smoothing_enable_checkbox.stateChanged.connect(self.on_smoothing_changed)

        smoothing_layout.addWidget(self.smoothing_label)
        smoothing_layout.addWidget(self.smoothing_method)
        smoothing_layout.addWidget(QLabel("LOWESS frac:", self))
        smoothing_layout.addWidget(self.lowess_edit)
        smoothing_layout.addWidget(QLabel("Savgol window:", self))
        smoothing_layout.addWidget(self.savgol_window_edit)
        smoothing_layout.addWidget(QLabel("polyorder:", self))
        smoothing_layout.addWidget(self.savgol_poly_edit)
        smoothing_layout.addWidget(self.smoothing_enable_checkbox)
        control_layout.addLayout(smoothing_layout)

        xviewrange_layout = QHBoxLayout()
        self.xviewrange_text = QLabel("View range")
        self.xviewrange = QRangeSlider(Qt.Horizontal)
        self.xviewrange.setEnabled(False)
        self.xviewrange.setFixedSize(800, 35)
        self.xviewrange.valueChanged.connect(self.update_xviewrange)
        xviewrange_layout.addWidget(self.xviewrange_text)
        xviewrange_layout.addWidget(self.xviewrange)
        control_layout.addLayout(xviewrange_layout)

        # Slider 1
        slider_layout1 = QHBoxLayout()
        self.sliderfit1_text = QLabel("Fit point 1:")
        self.sliderfit1 = QSlider(Qt.Horizontal)
        self.sliderfit1.setEnabled(False)
        self.sliderfit1.setFixedSize(500, 35)
        self.sliderfit1.valueChanged.connect(self.update_marker)
        self.sliderfit1_range_text = QLabel("Range of fit 1:")
        self.sliderfit1_range = QSlider(Qt.Horizontal)
        self.sliderfit1_range.setEnabled(False)
        self.sliderfit1_range.setFixedSize(200, 35)
        self.sliderfit1_range.valueChanged.connect(self.update_marker)
        slider_layout1.addWidget(self.sliderfit1_text)
        slider_layout1.addWidget(self.sliderfit1)
        slider_layout1.addWidget(self.sliderfit1_range_text)
        slider_layout1.addWidget(self.sliderfit1_range)
        control_layout.addLayout(slider_layout1)

        # Slider 2
        slider_layout2 = QHBoxLayout()
        self.sliderfit2_text = QLabel("Fit point 2:")
        self.sliderfit2 = QSlider(Qt.Horizontal)
        self.sliderfit2.setEnabled(False)
        self.sliderfit2.setFixedSize(500, 35)
        self.sliderfit2.valueChanged.connect(self.update_marker)
        self.sliderfit2_range_text = QLabel("Range of fit 2:")
        self.sliderfit2_range = QSlider(Qt.Horizontal)
        self.sliderfit2_range.setEnabled(False)
        self.sliderfit2_range.setFixedSize(200, 35)
        self.sliderfit2_range.valueChanged.connect(self.update_marker)
        slider_layout2.addWidget(self.sliderfit2_text)
        slider_layout2.addWidget(self.sliderfit2)
        slider_layout2.addWidget(self.sliderfit2_range_text)
        slider_layout2.addWidget(self.sliderfit2_range)
        control_layout.addLayout(slider_layout2)

        # Slider 3
        slider_layout3 = QHBoxLayout()
        self.sliderfit3_text = QLabel("Fit point 3:")
        self.sliderfit3 = QSlider(Qt.Horizontal)
        self.sliderfit3.setEnabled(False)
        self.sliderfit3.setFixedSize(500, 35)
        self.sliderfit3.valueChanged.connect(self.update_marker)
        self.sliderfit3_range_text = QLabel("Range of fit 3:")
        self.sliderfit3_range = QSlider(Qt.Horizontal)
        self.sliderfit3_range.setEnabled(False)
        self.sliderfit3_range.setFixedSize(200, 35)
        self.sliderfit3_range.valueChanged.connect(self.update_marker)
        slider_layout3.addWidget(self.sliderfit3_text)
        slider_layout3.addWidget(self.sliderfit3)
        slider_layout3.addWidget(self.sliderfit3_range_text)
        slider_layout3.addWidget(self.sliderfit3_range)
        control_layout.addLayout(slider_layout3)

        control_layout.addStretch()

        # Table setup
        self.lsv_result_display = pd.DataFrame(columns=['file name','E/I','1/I','E','I'])
        self.lsv_result_table = QtWidgets.QTableView()
        self.table_model = TableModel(self.lsv_result_display)
        self.lsv_result_table.setModel(self.table_model)
        self.copy_result_button = QPushButton("Copy results", self)
        self.copy_result_button.clicked.connect(self.copy_lsv_results)
        self.export_result_button = QPushButton("Export results", self)
        self.export_result_button.clicked.connect(self.export_lsv_results)
        table_button_layout = QHBoxLayout()
        table_button_layout.addStretch()
        table_button_layout.addWidget(self.copy_result_button)
        table_button_layout.addWidget(self.export_result_button)

        # Main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.plot_E_I, 1)
        top_layout.addWidget(self.plot_EV_I, 1)
        top_layout.addLayout(control_layout)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(table_button_layout)
        main_layout.addWidget(self.lsv_result_table)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Placeholders
        self.x = []
        self.y = []
        self.lsv = None
        self.marker = None
        self.df_combine_E = pd.DataFrame()
        self.df_combine_I = pd.DataFrame()
        self.lsv_idx = 0
        self.file_path_list = []
        self.file_name_list = []
        self.df_save_data = pd.DataFrame()
        self.fit_marker_E_I = None
        self.lsv_chosen_idx = 0

    def create_open_menu(self):
        add_lsv_menu = QMenu(self)
        autolab_action = QAction("AutoLab(.xlsx)", self)
        autolab_action.triggered.connect(lambda: self.open_file('AutoLab'))
        biologic_txt_action = QAction("Biologic(.txt)", self)
        biologic_txt_action.triggered.connect(lambda: self.open_file('Biologic(.txt)'))
        biologic_mpr_action = QAction("Biologic(.mpr)", self)
        biologic_mpr_action.triggered.connect(lambda: self.open_file('Biologic(.mpr)'))
        csv_action = QAction("CSV(.csv) 1st col is E and 2nd col is I", self)
        csv_action.triggered.connect(lambda: self.open_file('CSV'))
        add_lsv_menu.addAction(autolab_action)
        add_lsv_menu.addAction(biologic_txt_action)
        add_lsv_menu.addAction(biologic_mpr_action)
        add_lsv_menu.addAction(csv_action)
        return add_lsv_menu

    def open_file(self, file_type):
        if file_type == "AutoLab":
            multi_file_path, _ = QFileDialog.getOpenFileNames(self,"Open AutoLab File", "", f'{".xlsx"} Files (*{".xlsx"})')
            if not multi_file_path:
                return
            for file_path in multi_file_path:
                df = pd.read_excel(file_path)
                df = df[["WE(1).Potential (V)", "WE(1).Current (A)"]]
                df.columns = ["E", "I"]
                self.prepare_data(file_path,df)

        elif file_type == "Biologic(.txt)":
            multi_file_path, _ = QFileDialog.getOpenFileNames(self,"Open Biologic exported File", "", f'{".txt"} Files (*{".txt"})')
            if not multi_file_path:
                return
            for file_path in multi_file_path:
                df = pd.read_csv(file_path,sep="\t")
                df = df[["Ecell/V", "<I>/mA"]]
                df.columns = ["E", "I"]
                self.prepare_data(file_path,df)

        elif file_type == "Biologic(.mpr)":
            multi_file_path, _ = QFileDialog.getOpenFileNames(self,"Open Biologic File", "", f'{".mpr"} Files (*{".mpr"})')
            if not multi_file_path:
                return
            for file_path in multi_file_path:
                mpr_file = BioLogic.MPRfile(file_path)
                df = pd.DataFrame(mpr_file.data)
                df = df[["Ewe/V", "<I>/mA"]]
                df.columns = ["E", "I"]
                self.prepare_data(file_path,df)

        elif file_type == "CSV":
            multi_file_path, _ = QFileDialog.getOpenFileNames(self,"Open CSV File", "", f'{".csv"} Files (*{".csv"})')
            if not multi_file_path:
                return
            for file_path in multi_file_path:
                df = pd.read_csv(file_path)
                df.columns = ["E", "I"]
                self.prepare_data(file_path,df)

        self.df_E_max = max(self.df_combine_E.max())
        self.df_E_min = min(self.df_combine_E.min())
        self.df_I_max = max(self.df_combine_I.max())
        self.df_I_min = min(self.df_combine_I.min())
        self.df_E_max_idx = max(self.df_combine_E.index)
        self.lsv_num = self.df_combine_E.shape

        self.lsvchoosecombo.blockSignals(True)
        self.lsvchoosecombo.clear()
        self.lsvchoosecombo.addItems(self.file_name_list)
        self.lsvchoosecombo.setEnabled(True)
        last_idx = len(self.file_name_list) - 1
        if last_idx >= 0:
            self.lsvchoosecombo.setCurrentIndex(last_idx)
        self.lsvchoosecombo.blockSignals(False)
        self.delete_button.setEnabled(len(self.file_name_list) > 0)
        self.choose_lsv()

    def delete_selected_file(self):
        if not self.file_name_list:
            return

        idx = self.lsvchoosecombo.currentIndex()
        if idx < 0 or idx >= len(self.file_name_list):
            return

        del self.file_path_list[idx]
        del self.file_name_list[idx]

        if not self.df_combine_E.empty:
            self.df_combine_E.drop(columns=idx, inplace=True)
            self.df_combine_E.columns = range(self.df_combine_E.shape[1])
        if not self.df_combine_I.empty:
            self.df_combine_I.drop(columns=idx, inplace=True)
            self.df_combine_I.columns = range(self.df_combine_I.shape[1])
        if hasattr(self, 'df_combine_E_raw') and not self.df_combine_E_raw.empty:
            self.df_combine_E_raw.drop(columns=idx, inplace=True)
            self.df_combine_E_raw.columns = range(self.df_combine_E_raw.shape[1])
        if hasattr(self, 'df_combine_I_raw') and not self.df_combine_I_raw.empty:
            self.df_combine_I_raw.drop(columns=idx, inplace=True)
            self.df_combine_I_raw.columns = range(self.df_combine_I_raw.shape[1])

        self.lsv_result_display.drop(index=idx, inplace=True)
        self.lsv_result_display.reset_index(drop=True, inplace=True)
        self.df_save_data.drop(index=idx, inplace=True)
        self.df_save_data.reset_index(drop=True, inplace=True)
        self.lsv_result_table.setModel(TableModel(self.lsv_result_display))

        if not self.file_name_list:
            self.lsvchoosecombo.blockSignals(True)
            self.lsvchoosecombo.clear()
            self.lsvchoosecombo.setEnabled(False)
            self.lsvchoosecombo.blockSignals(False)
            self.delete_button.setEnabled(False)
            self.df_combine_E = pd.DataFrame()
            self.df_combine_I = pd.DataFrame()
            self.df_combine_E_raw = pd.DataFrame()
            self.df_combine_I_raw = pd.DataFrame()
            self.plot_E_I.clear()
            self.plot_EV_I.clear()
            self.fit_marker_E_I = None
            self.lsv_chosen_idx = 0
            self.disable_controls()
            return

        self.lsvchoosecombo.blockSignals(True)
        self.lsvchoosecombo.clear()
        self.lsvchoosecombo.addItems(self.file_name_list)
        self.lsvchoosecombo.setEnabled(True)
        new_idx = min(idx, len(self.file_name_list) - 1)
        self.lsvchoosecombo.setCurrentIndex(new_idx)
        self.lsvchoosecombo.blockSignals(False)
        self.delete_button.setEnabled(True)
        self.choose_lsv()

    def prepare_data(self, file_path, df):
        self.file_path_list.append(file_path)
        self.file_name_list.append(os.path.basename(file_path))
        self.lsv_null_result = pd.DataFrame({'file name': [os.path.basename(file_path)],
                                             'E/I': [np.nan], '1/I': [np.nan],
                                             'E': [np.nan],'I': [np.nan]})
        self.lsv_result_display = pd.concat([self.lsv_result_display,self.lsv_null_result],axis=0)

        self.df_save_null = pd.DataFrame({
            'file path': [file_path],
            'xviewrange start': [np.nan], 'xviewrange end': [np.nan],
            'slider1': [np.nan], 'range1': [np.nan],
            'slider2': [np.nan], 'range2': [np.nan],
            'slider3': [np.nan], 'range3': [np.nan],
            'lowess_frac': [0.1],
            'smoothing_enabled': [True],
            'smoothing_method': ["LOWESS"],
            'savgol_window': [5],
            'savgol_poly': [2]
        })
        self.df_save_data = pd.concat([self.df_save_data,self.df_save_null],axis=0)
        self.lsv_result_display.reset_index(drop=True, inplace=True)
        self.df_save_data.reset_index(drop=True, inplace=True)
        self.lsv_result_table.setModel(TableModel(self.lsv_result_display))

        # store plotting data
        self.df_combine_E = pd.concat([self.df_combine_E, df["E"]], axis=1)
        self.df_combine_I = pd.concat([self.df_combine_I, df["I"]], axis=1)
        self.df_combine_E.columns = range(self.df_combine_E.shape[1])
        self.df_combine_I.columns = range(self.df_combine_I.shape[1])

        # backup raw data for smoothing toggle
        if not hasattr(self, 'df_combine_E_raw'):
            self.df_combine_E_raw = pd.DataFrame()
            self.df_combine_I_raw = pd.DataFrame()
        self.df_combine_E_raw = pd.concat([self.df_combine_E_raw, df["E"]], axis=1)
        self.df_combine_I_raw = pd.concat([self.df_combine_I_raw, df["I"]], axis=1)
        self.df_combine_E_raw.columns = range(self.df_combine_E_raw.shape[1])
        self.df_combine_I_raw.columns = range(self.df_combine_I_raw.shape[1])

    def choose_lsv(self):
        self.lsv_chosen_name = self.lsvchoosecombo.currentText()
        self.lsv_chosen_idx = int(self.lsvchoosecombo.currentIndex())
        self.lsv_chosen_file_path = self.file_path_list[self.lsv_chosen_idx]

        # restore raw first
        self.E = self.df_combine_E_raw.iloc[:, self.lsv_chosen_idx].to_numpy()
        self.E = self.E[~np.isnan(self.E)]
        self.I = self.df_combine_I_raw.iloc[:, self.lsv_chosen_idx].to_numpy()
        self.I = self.I[~np.isnan(self.I)]

        frac = self.df_save_data.at[self.lsv_chosen_idx, 'lowess_frac']
        self.lowess_edit.blockSignals(True)
        self.lowess_edit.setText(f"{frac:.3f}")
        self.lowess_edit.blockSignals(False)

        self.smoothing_enable_checkbox.blockSignals(True)
        self.smoothing_enable_checkbox.setChecked(
            bool(self.df_save_data.at[self.lsv_chosen_idx, 'smoothing_enabled'])
        )
        self.smoothing_enable_checkbox.blockSignals(False)

        saved_method = self.df_save_data.at[self.lsv_chosen_idx, 'smoothing_method']
        self.smoothing_method.blockSignals(True)
        if saved_method in ["None", "LOWESS", "Savitzky-Golay"]:
            self.smoothing_method.setCurrentText(saved_method)
        else:
            self.smoothing_method.setCurrentText("None")
        self.smoothing_method.blockSignals(False)

        saved_window = int(self.df_save_data.at[self.lsv_chosen_idx, 'savgol_window'])
        saved_poly = int(self.df_save_data.at[self.lsv_chosen_idx, 'savgol_poly'])
        self.savgol_window_edit.blockSignals(True)
        self.savgol_poly_edit.blockSignals(True)
        self.savgol_window_edit.setText(str(saved_window))
        self.savgol_poly_edit.setText(str(saved_poly))
        self.savgol_window_edit.blockSignals(False)
        self.savgol_poly_edit.blockSignals(False)

        # apply smoothing if enabled
        self.I = self.apply_smoothing(self.E, self.I)

        self.lsv_chosen_size = len(self.E)
        self.EI = np.flip(self.E/self.I)
        self.invI = np.flip(1/self.I)

        self.config_slider()

    def _sanitize_lowess_frac(self):
        try:
            frac = float(self.lowess_edit.text())
        except ValueError:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        self.lowess_edit.blockSignals(True)
        self.lowess_edit.setText(f"{frac:.3f}")
        self.lowess_edit.blockSignals(False)
        return frac

    def _sanitize_savgol_params(self):
        try:
            window = int(self.savgol_window_edit.text())
        except (ValueError, TypeError):
            window = 5
        try:
            poly = int(self.savgol_poly_edit.text())
        except (ValueError, TypeError):
            poly = 2

        if window % 2 == 0:
            window += 1
        if window <= poly:
            window = poly + 2 + (poly % 2)
            if window % 2 == 0:
                window += 1

        self.savgol_window_edit.blockSignals(True)
        self.savgol_poly_edit.blockSignals(True)
        self.savgol_window_edit.setText(str(window))
        self.savgol_poly_edit.setText(str(poly))
        self.savgol_window_edit.blockSignals(False)
        self.savgol_poly_edit.blockSignals(False)
        return window, poly

    def apply_smoothing(self, E, I):
        if not self.smoothing_enable_checkbox.isChecked():
            return I

        method = self.smoothing_method.currentText()
        if method == "LOWESS":
            frac = self._sanitize_lowess_frac()
            self.df_save_data.at[self.lsv_chosen_idx, 'lowess_frac'] = frac
            if frac > 0:
                return smooth_current_lowess(E, I, frac)
            return I

        if method == "Savitzky-Golay":
            window, poly = self._sanitize_savgol_params()
            self.df_save_data.at[self.lsv_chosen_idx, 'savgol_window'] = window
            self.df_save_data.at[self.lsv_chosen_idx, 'savgol_poly'] = poly

            try:
                return savgol_filter(I, window_length=window, polyorder=poly)
            except Exception:
                return I

        return I

    def on_smoothing_changed(self):
        if not self.file_name_list:
            return

        frac = self._sanitize_lowess_frac()
        window, poly = self._sanitize_savgol_params()
        self.df_save_data.at[self.lsv_chosen_idx, 'smoothing_enabled'] = self.smoothing_enable_checkbox.isChecked()
        self.df_save_data.at[self.lsv_chosen_idx, 'smoothing_method'] = self.smoothing_method.currentText()
        self.df_save_data.at[self.lsv_chosen_idx, 'lowess_frac'] = frac
        self.df_save_data.at[self.lsv_chosen_idx, 'savgol_window'] = window
        self.df_save_data.at[self.lsv_chosen_idx, 'savgol_poly'] = poly
        self.choose_lsv()

    def disable_controls(self):
        sliders = [self.sliderfit1, self.sliderfit2, self.sliderfit3,
                   self.sliderfit1_range, self.sliderfit2_range, self.sliderfit3_range]
        for slider in sliders:
            slider.blockSignals(True)
            slider.setMinimum(0)
            slider.setMaximum(0)
            slider.setValue(0)
            slider.setEnabled(False)
            slider.blockSignals(False)

        self.xviewrange.blockSignals(True)
        self.xviewrange.setMinimum(0)
        self.xviewrange.setMaximum(0)
        self.xviewrange.setValue((0, 0))
        self.xviewrange.setEnabled(False)
        self.xviewrange.blockSignals(False)

    # ------------------------------
    # everything below is unchanged:
    # ------------------------------

    def config_slider(self):
        max_index = max(self.lsv_chosen_size - 1, 0)
        half_range = int(max_index / 2) if max_index > 0 else 0

        def clamp_value(value, lower, upper, default):
            try:
                if pd.isna(value):
                    return default
                numeric_val = int(value)
            except (TypeError, ValueError):
                return default
            return max(lower, min(upper, numeric_val))

        saved_slider1 = clamp_value(self.df_save_data.at[self.lsv_chosen_idx, 'slider1'], 0, max_index, 0)
        saved_slider2 = clamp_value(self.df_save_data.at[self.lsv_chosen_idx, 'slider2'], 0, max_index, 0)
        saved_slider3 = clamp_value(self.df_save_data.at[self.lsv_chosen_idx, 'slider3'], 0, max_index, 0)

        saved_range1 = clamp_value(self.df_save_data.at[self.lsv_chosen_idx, 'range1'], 0, half_range, 0)
        saved_range2 = clamp_value(self.df_save_data.at[self.lsv_chosen_idx, 'range2'], 0, half_range, 0)
        saved_range3 = clamp_value(self.df_save_data.at[self.lsv_chosen_idx, 'range3'], 0, half_range, 0)

        saved_x_start = clamp_value(self.df_save_data.at[self.lsv_chosen_idx, 'xviewrange start'], 0, max_index, 0)
        saved_x_end = clamp_value(self.df_save_data.at[self.lsv_chosen_idx, 'xviewrange end'], saved_x_start, max_index, max_index)

        self.sliderfit1.blockSignals(True)
        self.sliderfit1.setMinimum(0)
        self.sliderfit1.setMaximum(max_index)
        self.sliderfit1.setValue(saved_slider1)
        self.sliderfit1.setEnabled(True)
        self.sliderfit1.blockSignals(False)

        self.sliderfit2.blockSignals(True)
        self.sliderfit2.setMinimum(0)
        self.sliderfit2.setMaximum(max_index)
        self.sliderfit2.setValue(saved_slider2)
        self.sliderfit2.setEnabled(True)
        self.sliderfit2.blockSignals(False)

        self.sliderfit3.blockSignals(True)
        self.sliderfit3.setMinimum(0)
        self.sliderfit3.setMaximum(max_index)
        self.sliderfit3.setValue(saved_slider3)
        self.sliderfit3.setEnabled(True)
        self.sliderfit3.blockSignals(False)

        self.sliderfit1_range.blockSignals(True)
        self.sliderfit1_range.setMinimum(0)
        self.sliderfit1_range.setMaximum(half_range)
        self.sliderfit1_range.setValue(saved_range1)
        self.sliderfit1_range.setEnabled(True)
        self.sliderfit1_range.blockSignals(False)

        self.sliderfit2_range.blockSignals(True)
        self.sliderfit2_range.setMinimum(0)
        self.sliderfit2_range.setMaximum(half_range)
        self.sliderfit2_range.setValue(saved_range2)
        self.sliderfit2_range.setEnabled(True)
        self.sliderfit2_range.blockSignals(False)

        self.sliderfit3_range.blockSignals(True)
        self.sliderfit3_range.setMinimum(0)
        self.sliderfit3_range.setMaximum(half_range)
        self.sliderfit3_range.setValue(saved_range3)
        self.sliderfit3_range.setEnabled(True)
        self.sliderfit3_range.blockSignals(False)

        self.xviewrange.blockSignals(True)
        self.xviewrange.setMinimum(0)
        self.xviewrange.setMaximum(max_index)
        self.xviewrange.setValue((saved_x_start, saved_x_end))
        self.xviewrange.setEnabled(True)
        self.xviewrange.blockSignals(False)

        self.xviewrange_slider_start = saved_x_start
        self.xviewrange_slider_end = saved_x_end

        self.plot()
        self.update_xviewrange()
        self.update_marker()

    def plot_all_lsv(self):
        """Plot all CVs, highlighting the selected one (smoothed if enabled)."""
        grey_pen = pg.mkPen('gray', width=1, style=QtCore.Qt.DotLine)
        red_pen = pg.mkPen('red', width=1)

        for i in range(self.df_combine_E.shape[1]):
            E_series = self.df_combine_E_raw[i]
            I_series = self.df_combine_I_raw[i]

            E_values = E_series.to_numpy()
            I_values = I_series.to_numpy()
            valid_mask = ~np.isnan(E_values) & ~np.isnan(I_values)
            E_values = E_values[valid_mask]
            I_values = I_values[valid_mask]

            if i == self.lsv_chosen_idx:
                plot_E = self.E
                plot_I = self.I
                pen_left = red_pen
                pen_right = red_pen
            else:
                plot_E = E_values
                plot_I = I_values
                pen_left = grey_pen
                pen_right = grey_pen

            if len(plot_E) == 0 or len(plot_I) == 0:
                continue

            with np.errstate(divide='ignore', invalid='ignore'):
                inv_I = 1 / plot_I
                E_over_I = plot_E / plot_I

            finite_mask = np.isfinite(inv_I) & np.isfinite(E_over_I)
            inv_I = inv_I[finite_mask]
            E_over_I = E_over_I[finite_mask]

            finite_mask_left = np.isfinite(plot_E) & np.isfinite(plot_I)
            plot_E_left = plot_E[finite_mask_left]
            plot_I_left = plot_I[finite_mask_left]

            if len(plot_E_left) == 0 or len(plot_I_left) == 0:
                continue

            self.plot_E_I.plot(plot_E_left, plot_I_left, pen=pen_left)

            if len(inv_I) > 0 and len(E_over_I) > 0:
                self.plot_EV_I.plot(inv_I, E_over_I, pen=pen_right)


    def plot(self):
        self.plot_E_I.clear()
        self.plot_EV_I.clear()
        self.plot_all_lsv()
        self.slider_marker_fit1 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='red',symbolSize=8)
        self.slider_marker_fit2 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='blue',symbolSize=8)
        self.slider_marker_fit3 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='darkgreen',symbolSize=8)
        self.slider_marker_fit1_range_start = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='red',symbolSize=13)
        self.slider_marker_fit2_range_start = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='blue',symbolSize=13)
        self.slider_marker_fit3_range_start = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='darkgreen',symbolSize=13)
        self.slider_marker_fit1_range_end = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='red',symbolSize=13)
        self.slider_marker_fit2_range_end = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='blue',symbolSize=13)
        self.slider_marker_fit3_range_end = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='darkgreen',symbolSize=13)

        self.datafit1 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('red', width=1.5, style=QtCore.Qt.DashLine))
        self.datafit2 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('blue', width=1.5, style=QtCore.Qt.DashLine))
        self.datafit3 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('blue', width=1.5, style=QtCore.Qt.DashLine))
        self.datafit4 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('darkgreen', width=1.5, style=QtCore.Qt.DashLine))

        self.point1 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='red',symbolSize=8)
        self.point2 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='darkgreen',symbolSize=8)
        self.midpoint1 = self.plot_EV_I.plot([0],[0],pen=None,symbol='x',symbolBrush='blue',symbolSize=13)
        self.midpointE1 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('indianred', width=1.5, style=QtCore.Qt.DashLine))

        if self.fit_marker_E_I is None:
            self.fit_marker_E_I = self.plot_E_I.plot([], [], pen=None, symbol='x',
                                                    symbolBrush='indianred',
                                                    symbolPen=pg.mkPen('indianred'),
                                                    symbolSize=12)
        else:
            self.fit_marker_E_I = self.plot_E_I.plot([], [], pen=None, symbol='x',
                                                    symbolBrush='indianred',
                                                    symbolPen=pg.mkPen('indianred'),
                                                    symbolSize=12)

        current_E = self.lsv_result_display.at[self.lsv_chosen_idx, 'E']
        current_I = self.lsv_result_display.at[self.lsv_chosen_idx, 'I']
        if pd.notna(current_E) and pd.notna(current_I):
            self.fit_marker_E_I.setData([current_E], [current_I])

    def update_xviewrange(self):
        self.xviewrange_slider_start = self.xviewrange.value()[0]
        self.xviewrange_slider_end = self.xviewrange.value()[1]
        self.low_xviewrange = self.invI[self.xviewrange_slider_start]
        self.high_xviewrange = self.invI[self.xviewrange_slider_end]

        if self.EI[self.xviewrange_slider_start] >= np.min(self.EI[self.xviewrange_slider_start:self.xviewrange_slider_end]):
            self.low_yviewrange = np.min(self.EI[self.xviewrange_slider_start:self.xviewrange_slider_end])
        else:
            self.low_yviewrange = self.EI[self.xviewrange.value()[0]]

        if self.EI[self.xviewrange_slider_end] <= np.max(self.EI[self.xviewrange_slider_start:self.xviewrange_slider_end]):
            self.high_yviewrange = np.max(self.EI[self.xviewrange_slider_start:self.xviewrange_slider_end])
        else:
            self.high_yviewrange = self.EI[self.xviewrange.value()[1]]

        self.plot_EV_I.setXRange(self.low_xviewrange,self.high_xviewrange,padding=0)
        self.plot_EV_I.setYRange(self.low_yviewrange,self.high_yviewrange,padding=0.2)
        self.save_data()

    def update_marker(self):
        start1 = self.sliderfit1.value()-self.sliderfit1_range.value()
        end1 = self.sliderfit1.value()+self.sliderfit1_range.value()
        start2 = self.sliderfit2.value()-self.sliderfit2_range.value()
        end2 = self.sliderfit2.value()+self.sliderfit2_range.value()
        start3 = self.sliderfit3.value()-self.sliderfit3_range.value()
        end3 = self.sliderfit3.value()+self.sliderfit3_range.value()

        self.slider_marker_fit1_range_start.setData([self.invI[start1]],[self.EI[start1]])
        self.slider_marker_fit1_range_end.setData([self.invI[end1]],[self.EI[end1]])
        self.slider_marker_fit2_range_start.setData([self.invI[start2]],[self.EI[start2]])
        self.slider_marker_fit2_range_end.setData([self.invI[end2]],[self.EI[end2]])
        self.slider_marker_fit3_range_start.setData([self.invI[start3]],[self.EI[start3]])
        self.slider_marker_fit3_range_end.setData([self.invI[end3]],[self.EI[end3]])

        lnfit1 = sorted([start1,end1,start2,end2])
        lnfit2 = sorted([start2,end2,start3,end3])

        try:
            coeff1 = np.polyfit(self.invI[start1:end1], self.EI[start1:end1], 1)
            poly1 = np.poly1d(coeff1)
            y_fit1 = poly1(self.invI[lnfit1[0]:lnfit1[-1]])
            self.datafit1.setData(self.invI[lnfit1[0]:lnfit1[-1]],y_fit1)
        except TypeError:
            pass

        try:
            coeff2 = np.polyfit(self.invI[start2:end2], self.EI[start2:end2], 1)
            poly2 = np.poly1d(coeff2)
            y_fit2 = poly2(self.invI[lnfit1[0]:lnfit2[-1]])
            self.datafit2.setData(self.invI[lnfit1[0]:lnfit2[-1]],y_fit2)
        except TypeError:
            pass

        try:
            coeff3 = np.polyfit(self.invI[start3:end3], self.EI[start3:end3], 1)
            poly3 = np.poly1d(coeff3)
            y_fit3 = poly3(self.invI[lnfit2[0]:lnfit2[-1]])
            self.datafit4.setData(self.invI[lnfit2[0]:lnfit2[-1]],y_fit3)
        except TypeError:
            pass

        try:
            x1 = (coeff2[1]-coeff1[1])/(coeff1[0]-coeff2[0])
            y1 = poly1(x1)
            self.point1.setData([x1],[y1])
        except UnboundLocalError:
            pass

        try:
            x2 = (coeff3[1]-coeff2[1])/(coeff2[0]-coeff3[0])
            y2 = poly2(x2)
            self.point2.setData([x2],[y2])
        except UnboundLocalError:
            pass

        try:
            xmid = (x1+x2)/2
            ymid = (y1+y2)/2
            padmidpointy1 = ((self.high_yviewrange+self.low_yviewrange)/2)*0.2
            self.midpoint1.setData([xmid],[ymid])
            self.midpointE1.setData([xmid,xmid],[self.low_yviewrange-padmidpointy1,ymid])

            self.lsv_result_display.at[self.lsv_chosen_idx, 'E/I'] = ymid
            self.lsv_result_display.at[self.lsv_chosen_idx, '1/I'] = xmid
            self.lsv_result_display.at[self.lsv_chosen_idx, 'E'] = ymid/xmid
            self.lsv_result_display.at[self.lsv_chosen_idx, 'I'] = 1/xmid
            self.lsv_result_table.setModel(TableModel(self.lsv_result_display))
            if self.fit_marker_E_I is not None:
                self.fit_marker_E_I.setData([self.lsv_result_display.at[self.lsv_chosen_idx, 'E']],
                                            [self.lsv_result_display.at[self.lsv_chosen_idx, 'I']])
        except UnboundLocalError:
            if self.fit_marker_E_I is not None:
                self.fit_marker_E_I.setData([], [])

        self.save_data()

    def save_data(self):
        self.df_save_data.at[self.lsv_chosen_idx, 'xviewrange start'] = self.xviewrange_slider_start
        self.df_save_data.at[self.lsv_chosen_idx, 'xviewrange end'] = self.xviewrange_slider_end
        self.df_save_data.at[self.lsv_chosen_idx, 'slider1'] = self.sliderfit1.value()
        self.df_save_data.at[self.lsv_chosen_idx, 'slider2'] = self.sliderfit2.value()
        self.df_save_data.at[self.lsv_chosen_idx, 'slider3'] = self.sliderfit3.value()
        self.df_save_data.at[self.lsv_chosen_idx, 'range1'] = self.sliderfit1_range.value()
        self.df_save_data.at[self.lsv_chosen_idx, 'range2'] = self.sliderfit2_range.value()
        self.df_save_data.at[self.lsv_chosen_idx, 'range3'] = self.sliderfit3_range.value()
        self.df_save_data.at[self.lsv_chosen_idx, 'smoothing_enabled'] = self.smoothing_enable_checkbox.isChecked()
        self.df_save_data.at[self.lsv_chosen_idx, 'smoothing_method'] = self.smoothing_method.currentText()

    def copy_lsv_results(self):
        if self.lsv_result_display.empty:
            return

        copy_text = self.lsv_result_display.to_csv(sep='\t', index=False, na_rep='')
        QtWidgets.QApplication.clipboard().setText(copy_text)

    def export_lsv_results(self):
        if self.lsv_result_display.empty:
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export LSV results",
            "lsv_results.xlsx",
            "Excel Workbook (*.xlsx);;OpenDocument Spreadsheet (*.ods)"
        )
        if not file_path:
            return

        if selected_filter == "OpenDocument Spreadsheet (*.ods)" and not file_path.lower().endswith('.ods'):
            file_path += '.ods'
        elif selected_filter == "Excel Workbook (*.xlsx)" and not file_path.lower().endswith('.xlsx'):
            file_path += '.xlsx'

        try:
            self.lsv_result_display.to_excel(file_path, index=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", f"Could not export file:\n{exc}")
            return

        QtWidgets.QMessageBox.information(self, "Export results", f"Results exported to:\n{file_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())

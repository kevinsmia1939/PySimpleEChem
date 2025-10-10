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

        # Plot widget
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

        self.lsvchoosecombo = QComboBox(self)
        self.lsvchoosecombo.setFixedSize(300, 35)
        self.lsvchoosecombo.setEditable(False)
        self.lsvchoosecombo.setEnabled(False)
        self.lsvchoosecombo.currentIndexChanged.connect(self.choose_lsv)

        open_button_layout.addWidget(self.open_button)
        open_button_layout.addWidget(self.lsvchoosecombo)
        control_layout.addLayout(open_button_layout)

        # LOWESS smoothing UI
        lowess_layout = QHBoxLayout()
        self.lowess_label = QLabel("Smoothing (LOWESS frac):", self)
        self.lowess_edit = QtWidgets.QLineEdit("0.0", self)
        self.lowess_edit.setFixedWidth(80)
        self.lowess_edit.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 3))

        self.lowess_checkbox = QCheckBox("Enable")
        self.lowess_checkbox.setChecked(False)

        self.lowess_edit.editingFinished.connect(self.on_lowess_frac_changed)
        self.lowess_checkbox.stateChanged.connect(self.on_lowess_frac_changed)

        lowess_layout.addWidget(self.lowess_label)
        lowess_layout.addWidget(self.lowess_edit)
        lowess_layout.addWidget(self.lowess_checkbox)
        control_layout.addLayout(lowess_layout)

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

        # Main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.plot_EV_I, 1)
        top_layout.addLayout(control_layout)
        main_layout.addLayout(top_layout)
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
            'lowess_frac': [0.0]
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
        self.lowess_edit.setText(f"{frac:.3f}")

        # apply smoothing if enabled
        if self.lowess_checkbox.isChecked() and frac > 0:
            self.I = smooth_current_lowess(self.E, self.I, frac)

        self.lsv_chosen_size = len(self.E)
        self.EI = np.flip(self.E/self.I)
        self.invI = np.flip(1/self.I)

        self.config_slider()

    def on_lowess_frac_changed(self):
        try:
            frac = float(self.lowess_edit.text())
            frac = max(0.0, min(1.0, frac))
        except ValueError:
            frac = 0.0
            self.lowess_edit.setText("0.0")
        self.df_save_data.at[self.lsv_chosen_idx, 'lowess_frac'] = frac
        self.choose_lsv()

    # ------------------------------
    # everything below is unchanged:
    # ------------------------------

    def config_slider(self):
        self.sliderfit1.blockSignals(True)
        self.sliderfit1.setMinimum(0)
        self.sliderfit1.setMaximum(self.lsv_chosen_size-1)
        self.sliderfit1.setValue(0)
        self.sliderfit1.setEnabled(True)
        self.sliderfit1.blockSignals(False)

        self.sliderfit2.blockSignals(True)
        self.sliderfit2.setMinimum(0)
        self.sliderfit2.setMaximum(self.lsv_chosen_size-1)
        self.sliderfit2.setValue(0)
        self.sliderfit2.setEnabled(True)
        self.sliderfit2.blockSignals(False)

        self.sliderfit3.blockSignals(True)
        self.sliderfit3.setMinimum(0)
        self.sliderfit3.setMaximum(self.lsv_chosen_size-1)
        self.sliderfit3.setValue(0)
        self.sliderfit3.setEnabled(True)
        self.sliderfit3.blockSignals(False)

        self.sliderfit1_range.blockSignals(True)
        self.sliderfit1_range.setMinimum(0)
        self.sliderfit1_range.setMaximum(int((self.lsv_chosen_size-1)/2))
        self.sliderfit1_range.setValue(0)
        self.sliderfit1_range.setEnabled(True)
        self.sliderfit1_range.blockSignals(False)

        self.sliderfit2_range.blockSignals(True)
        self.sliderfit2_range.setMinimum(0)
        self.sliderfit2_range.setMaximum(int((self.lsv_chosen_size-1)/2))
        self.sliderfit2_range.setValue(0)
        self.sliderfit2_range.setEnabled(True)
        self.sliderfit2_range.blockSignals(False)

        self.sliderfit3_range.blockSignals(True)
        self.sliderfit3_range.setMinimum(0)
        self.sliderfit3_range.setMaximum(int((self.lsv_chosen_size-1)/2))
        self.sliderfit3_range.setValue(0)
        self.sliderfit3_range.setEnabled(True)
        self.sliderfit3_range.blockSignals(False)

        self.xviewrange.blockSignals(True)
        self.xviewrange.setMinimum(0)
        self.xviewrange.setMaximum(self.lsv_chosen_size-1)
        self.xviewrange.setValue((0,self.lsv_chosen_size-1))
        self.xviewrange.setEnabled(True)
        self.xviewrange.blockSignals(False)

        self.plot()

    def plot_all_lsv(self):
        """Plot all CVs, and highlight the currently selected one (smoothed if enabled)."""
        for i in range(self.df_combine_E.shape[1]):
            E = self.df_combine_E_raw[i]  # always raw backup for others
            I = self.df_combine_I_raw[i]
            if i == self.lsv_chosen_idx:
                # Plot selected curve using current self.E/self.I (smoothed if enabled)
                self.lsv = self.plot_EV_I.plot(1/self.I, self.E/self.I, pen=pg.mkPen('red', width=1))
            else:
                self.plot_EV_I.plot(1/I, E/I, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DotLine))


    def plot(self):
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
        except UnboundLocalError:
            pass

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())

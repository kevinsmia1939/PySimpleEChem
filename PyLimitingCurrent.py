#!/usr/bin/python3
import sys
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QHBoxLayout, QTableView, QComboBox, QSlider, QLabel,
    QPushButton, QFileDialog
)
from PyQt5.QtCore import Qt
from superqt import QRangeSlider
from function_collection import (
    battery_xls2df, get_CV_init, cy_idx_state_range,
    read_cv_format, get_peak_CV, search_pattern, ir_compen_func,
    diffusion, reaction_rate, peak_2nd_deriv, find_alpha,
    min_max_peak, check_val, switch_val, RDE_kou_lev,
    linear_fit, data_poly_inter, open_battery_data,
    df_select_column, read_cv_versastat
)

pg.setConfigOption('background', 'white')
pg.setConfigOption('antialias', True)


class TableModel(QtCore.QAbstractTableModel):
    # https://www.pythonguis.com/tutorials/qtableview-modelviews-numpy-pandas/
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
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_file)

        self.lsvchoosecombo = QComboBox(self)
        self.lsvchoosecombo.setFixedSize(300, 35)
        self.lsvchoosecombo.setEditable(False)
        self.lsvchoosecombo.setInsertPolicy(QComboBox.NoInsert)
        self.lsvchoosecombo.setEnabled(False)
        self.lsvchoosecombo.currentIndexChanged.connect(self.choose_lsv)
        
        open_button_layout.addWidget(self.open_button)
        open_button_layout.addWidget(self.lsvchoosecombo)
        control_layout.addLayout(open_button_layout)

        xviewrange_layout = QHBoxLayout()
        self.xviewrange_text = QLabel("View range")
        self.xviewrange = QRangeSlider(Qt.Horizontal)
        self.xviewrange.setEnabled(False)
        self.xviewrange.setFixedSize(800, 35)
        self.xviewrange.valueChanged.connect(self.update_xviewrange)
        xviewrange_layout.addWidget(self.xviewrange_text)
        xviewrange_layout.addWidget(self.xviewrange)
        control_layout.addLayout(xviewrange_layout)

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
        # top: plot + controls
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.plot_EV_I, 1)
        top_layout.addLayout(control_layout)
        main_layout.addLayout(top_layout)
        # bottom: table
        main_layout.addWidget(self.lsv_result_table)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Placeholders
        self.x = []
        self.y = []
        self.lsv = None
        self.marker = None
        self.df_combine_E = pd.DataFrame()  # dataframe for potential
        self.df_combine_I = pd.DataFrame()  # dataframe for current
        self.lsv_idx = 0  # chosen lsv data index
        self.file_path_list = []
        self.file_name_list = []
        self.df_save_data = pd.DataFrame() #Save information on the file and slider settings 

    def open_file(self):
        multi_file_path, _ = QFileDialog.getOpenFileNames(self,"Open Data File","","Data Files (*.xlsx *.xls *.ods *.csv *.txt)")
        if not multi_file_path:
            return
        # Read data into DataFrame
        for file_path in multi_file_path:
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext in ['.xlsx', '.xls', '.ods']:
                    df = pd.read_excel(file_path)
                    df.columns = ['E', 'I']
                elif ext in ['.csv', '.txt']:
                    with open(file_path, 'r', newline='') as f:
                        sample = f.read(1024)
                        dialect = csv.Sniffer().sniff(sample)
                        delimiter = dialect.delimiter
                    df = pd.read_csv(file_path, delimiter=delimiter)
                    df.columns = ['E', 'I']
                else:
                    raise ValueError(f"Unsupported file type: {ext}")

                df_numeric = df.select_dtypes(include=["number"])
                is_numeric_df = df.shape[1] == df_numeric.shape[1]
                if not is_numeric_df:
                    raise ValueError(file_path, "contain non numerical value")
                if df.shape[1] != 2:
                    raise ValueError(file_path, "Data must have 2 columns with first column as E and second column as I")
                self.file_path_list.append(file_path)
                self.file_name_list.append(os.path.basename(file_path))
                self.lsv_null_result = pd.DataFrame({'file name': [os.path.basename(file_path)], 'E/I': [np.nan], '1/I': [np.nan], 'E': [np.nan],'I': [np.nan]})
                self.lsv_result_display = pd.concat([self.lsv_result_display,self.lsv_null_result],axis=0)
                
            except Exception as e:
                print(f"Error reading file: {e}")
                continue
           
            #update table
            self.lsv_result_display.reset_index(drop=True, inplace=True)

            self.lsv_result_table.setModel(TableModel(self.lsv_result_display))
            
            self.df_combine_E = pd.concat([self.df_combine_E, df["E"]], axis=1)
            self.df_combine_I = pd.concat([self.df_combine_I, df["I"]], axis=1)
        print(self.lsv_result_display)
        self.df_E_max = max(self.df_combine_E.max())
        self.df_E_min = min(self.df_combine_E.min())
        self.df_I_max = max(self.df_combine_I.max())
        self.df_I_min = min(self.df_combine_I.min())
        self.df_E_max_idx = max(self.df_combine_E.index)
        self.lsv_num = self.df_combine_E.shape
        
        # Set combo, this has to be done at the beginning
        self.lsvchoosecombo.blockSignals(True)
        self.lsvchoosecombo.clear()
        self.lsvchoosecombo.addItems(self.file_name_list)
        self.lsvchoosecombo.setEnabled(True)
        self.lsvchoosecombo.blockSignals(False)
        self.choose_lsv()
        
    def choose_lsv(self):
        self.lsv_chosen_name = self.lsvchoosecombo.currentText()
        self.lsv_chosen_idx = int(self.lsvchoosecombo.currentIndex())
        self.lsv_chosen_file_path = self.file_path_list[self.lsv_chosen_idx]

        self.E = self.df_combine_E.iloc[:, self.lsv_chosen_idx].to_numpy()
        self.E = self.E[~np.isnan(self.E)]
        self.I = self.df_combine_I.iloc[:, self.lsv_chosen_idx].to_numpy()
        self.I = self.I[~np.isnan(self.I)]
        self.lsv_chosen_size = len(self.E)
        self.EI = np.flip(self.E/self.I)
        self.invI = np.flip(1/self.I)
        print(self.lsv_chosen_idx,self.lsv_chosen_name,self.lsv_chosen_size,self.lsv_chosen_file_path)
        self.config_slider()

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

    def plot(self):
        self.plot_EV_I.clear()
        self.lsv = self.plot_EV_I.plot(self.invI,self.EI, pen=pg.mkPen('black', width=1.5))
        self.slider_marker_fit1 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='r',symbolSize=8)
        self.slider_marker_fit2 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='b',symbolSize=8)
        self.slider_marker_fit3 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='g',symbolSize=8)
        self.slider_marker_fit1_range_start = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='r',symbolSize=13)
        self.slider_marker_fit2_range_start = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='b',symbolSize=13)
        self.slider_marker_fit3_range_start = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='g',symbolSize=13)
        self.slider_marker_fit1_range_end = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='r',symbolSize=13)
        self.slider_marker_fit2_range_end = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='b',symbolSize=13)
        self.slider_marker_fit3_range_end = self.plot_EV_I.plot([0],[0],pen=None,symbol='d',symbolBrush='g',symbolSize=13)
        
        self.datafit1 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.DashLine))
        self.datafit2 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('b', width=1.5, style=QtCore.Qt.DashLine))
        self.datafit3 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('b', width=1.5, style=QtCore.Qt.DashLine))
        self.datafit4 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('g', width=1.5, style=QtCore.Qt.DashLine))

        self.point1 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='r',symbolSize=8)
        self.point2 = self.plot_EV_I.plot([0],[0],pen=None,symbol='o',symbolBrush='g',symbolSize=8)
        self.midpoint1 = self.plot_EV_I.plot([0],[0],pen=None,symbol='x',symbolBrush='b',symbolSize=13)
        self.midpointE1 = self.plot_EV_I.plot([0],[0], pen=pg.mkPen('indianred', width=1.5, style=QtCore.Qt.DashLine))
        
    def update_xviewrange(self):
        xviewrange_slider_start = self.xviewrange.value()[0]
        xviewrange_slider_end   = self.xviewrange.value()[1]
        self.low_xviewrange  = self.invI[xviewrange_slider_start]
        self.high_xviewrange  = self.invI[xviewrange_slider_end]
        if self.EI[xviewrange_slider_start] >= np.min(self.EI[xviewrange_slider_start:xviewrange_slider_end]):
            self.low_yviewrange  = np.min(self.EI[xviewrange_slider_start:xviewrange_slider_end])
        else:
            self.low_yviewrange  = self.EI[self.xviewrange.value()[0]]
            
        if self.EI[xviewrange_slider_end] <= np.max(self.EI[xviewrange_slider_start:xviewrange_slider_end]):
            self.high_yviewrange  = np.max(self.EI[xviewrange_slider_start:xviewrange_slider_end])
        else:
            self.high_yviewrange  = self.EI[self.xviewrange.value()[1]]        
            
        self.plot_EV_I.setXRange(self.low_xviewrange,self.high_xviewrange,padding=0)
        self.plot_EV_I.setYRange(self.low_yviewrange,self.high_yviewrange,padding=0.2)


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

        #Draw fitted line
        try: #incase start1:end1 is empty
            coeff1 = np.polyfit(self.invI[start1:end1], self.EI[start1:end1], 1)
            poly1 = np.poly1d(coeff1)
            y_fit1 = poly1(self.invI[lnfit1[0]:lnfit1[-1]])
            self.datafit1.setData(self.invI[lnfit1[0]:lnfit1[-1]],y_fit1)
            # print(coeff1)
        except TypeError:
            pass  
        
        try:
            coeff2 = np.polyfit(self.invI[start2:end2], self.EI[start2:end2], 1)
            poly2 = np.poly1d(coeff2)
            y_fit2 = poly2(self.invI[lnfit1[0]:lnfit2[-1]])
            self.datafit2.setData(self.invI[lnfit1[0]:lnfit2[-1]],y_fit2)
            # print(poly1)
        except TypeError:
            pass

        try:
            coeff3 = np.polyfit(self.invI[start3:end3], self.EI[start3:end3], 1)
            poly3 = np.poly1d(coeff3)
            y_fit3 = poly3(self.invI[lnfit2[0]:lnfit2[-1]])
            self.datafit4.setData(self.invI[lnfit2[0]:lnfit2[-1]],y_fit3)
            # print(poly3)
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
            
            self.lsv_result_display.at[self.lsv_chosen_idx, 'E/I']  = ymid
            self.lsv_result_display.at[self.lsv_chosen_idx, '1/I']  = xmid
            self.lsv_result_display.at[self.lsv_chosen_idx, 'E']    = ymid/xmid
            self.lsv_result_display.at[self.lsv_chosen_idx, 'I']    = 1/xmid           
            self.lsv_result_table.setModel(TableModel(self.lsv_result_display))
            
        except UnboundLocalError:
            pass      
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

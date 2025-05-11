#!/usr/bin/python3
import sys
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QPushButton, QFileDialog, QDialog, QHBoxLayout, QGridLayout, QComboBox, QLineEdit, QScrollArea, QTableWidget, QTableWidgetItem, QFrame, QCheckBox, QMenu, QAction, QSplitter, QTabWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from function_collection import battery_xls2df, get_CV_init, cy_idx_state_range, read_cv_format, get_peak_CV, search_pattern, ir_compen_func, diffusion, reaction_rate, peak_2nd_deriv, find_alpha, min_max_peak, check_val, switch_val, RDE_kou_lev, linear_fit, data_poly_inter,open_battery_data, df_select_column, read_cv_versastat
from superqt import QRangeSlider

pg.setConfigOption('background', 'white')
pg.setConfigOption('antialias', True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Plotter")
        control_layout = QVBoxLayout()
        # Plot widget
        self.plot_EV_I = pg.PlotWidget()
        self.plot_EV_I.setLabel('left', text='E/I')
        self.plot_EV_I.setLabel('bottom', text='1/I')
        self.plot_EV_I.getAxis('bottom').setTextPen('black')
        self.plot_EV_I.getAxis('left').setTextPen('black')
        
        # Controls: Open button and slider
        open_button_layout =  QHBoxLayout()
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_file)
        
        self.lsvchoosecombo = QComboBox(self)
        self.lsvchoosecombo.setFixedSize(300, 35)
        self.lsvchoosecombo.setEditable(False)
        self.lsvchoosecombo.setInsertPolicy(QComboBox.NoInsert)
        self.lsvchoosecombo.setEnabled(False)
        self.lsvchoosecombo.textActivated.connect(self.choose_lsv)
        
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

        # Main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.plot_EV_I, 1)
        main_layout.addLayout(control_layout)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Placeholders
        self.x = []
        self.y = []
        self.curve = None
        self.marker = None
        self.df_combine_E = pd.DataFrame() #dataframe for potential
        self.df_combine_I = pd.DataFrame() #dataframe for current
        self.lsv_idx = 0 #chosen lsv data index
        self.filepath_list = []
        self.filename_list = []
    def open_file(self):
        multi_path, _ = QFileDialog.getOpenFileNames(self,
            "Open Data File",
            "",
            "Data Files (*.xlsx *.xls *.ods *.csv *.txt)"
        )
        if not multi_path:
            return
        # Read data into DataFrame
        for path in multi_path:
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext in ['.xlsx', '.xls', '.ods']:
                    df = pd.read_excel(path)
                    df.columns = ['E', 'I']
                elif ext in ['.csv', '.txt']:
                    # Auto-detect delimiter
                    with open(path, 'r', newline='') as f:
                        sample = f.read(1024)
                        dialect = csv.Sniffer().sniff(sample)
                        delimiter = dialect.delimiter
                    df = pd.read_csv(path, delimiter=delimiter)
                    df.columns = ['E', 'I']
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
                
                # Verify if the DataFrame is fully numeric
                df_numeric = df.select_dtypes(include=["number"])
                is_numeric_df = df.shape[1] == df_numeric.shape[1]
                if is_numeric_df == False:
                    raise ValueError(path," contain non numerical value")
                if df.shape[1] != 2:
                    raise ValueError(path, "Data must have 2 columns with first column as E and second column as I")
                self.filepath_list.append(path) #store all chosen data path
            except Exception as e:
                print(f"Error reading file: {e}")
                pass
            self.df_combine_E = pd.concat([self.df_combine_E,df["E"]],axis=1)
            self.df_combine_I = pd.concat([self.df_combine_I,df["I"]],axis=1)
            
        for i in self.filepath_list: #get all lsv name
            self.filename_list.append(os.path.basename(i))
        self.df_E_max = max(self.df_combine_E.max())
        self.df_E_min = min(self.df_combine_E.min())
        self.df_I_max = max(self.df_combine_I.max())
        self.df_I_min = min(self.df_combine_I.min())
        self.df_E_max_idx = max(self.df_combine_E.index) #Max voltage in all LSVs
        self.lsv_num = self.df_combine_E.shape
        
        self.config_slider()
        # self.update_xviewrange()

        
    def config_slider(self):
        # Configure slider
        self.sliderfit1.blockSignals(True)
        self.sliderfit1.setMinimum(0)
        self.sliderfit1.setMaximum(self.df_E_max_idx-1)
        self.sliderfit1.setValue(0)
        self.sliderfit1.setEnabled(True)
        self.sliderfit1.blockSignals(False)
        ########################################################
        self.sliderfit2.blockSignals(True)
        self.sliderfit2.setMinimum(0)
        self.sliderfit2.setMaximum(self.df_E_max_idx-1)
        self.sliderfit2.setValue(0)
        self.sliderfit2.setEnabled(True)
        self.sliderfit2.blockSignals(False)
        ########################################################
        self.sliderfit3.blockSignals(True)
        self.sliderfit3.setMinimum(0)
        self.sliderfit3.setMaximum(self.df_E_max_idx-1)
        self.sliderfit3.setValue(0)
        self.sliderfit3.setEnabled(True)
        self.sliderfit3.blockSignals(False)
        ########################################################
        self.sliderfit1_range.blockSignals(True)
        self.sliderfit1_range.setMinimum(0)
        self.sliderfit1_range.setMaximum(int((self.df_E_max_idx-1)/2))
        self.sliderfit1_range.setValue(0)
        self.sliderfit1_range.setEnabled(True)
        self.sliderfit1_range.blockSignals(False)
        ########################################################
        self.sliderfit2_range.blockSignals(True)
        self.sliderfit2_range.setMinimum(0)
        self.sliderfit2_range.setMaximum(int((self.df_E_max_idx-1)/2))
        self.sliderfit2_range.setValue(0)
        self.sliderfit2_range.setEnabled(True)
        self.sliderfit2_range.blockSignals(False)
        ########################################################
        self.sliderfit3_range.blockSignals(True)
        self.sliderfit3_range.setMinimum(0)
        self.sliderfit3_range.setMaximum(int((self.df_E_max_idx-1)/2))
        self.sliderfit3_range.setValue(0)
        self.sliderfit3_range.setEnabled(True)
        self.sliderfit3_range.blockSignals(False)
        ########################################################
        self.xviewrange.blockSignals(True)
        self.xviewrange.setMinimum(0)
        self.xviewrange.setMaximum(self.df_E_max_idx-1)
        self.xviewrange.setValue((0,self.df_E_max_idx-1))
        self.xviewrange.setEnabled(True)
        self.xviewrange.blockSignals(False)  
        ########################################################
        self.lsvchoosecombo.blockSignals(True)
        self.lsvchoosecombo.setEnabled(True)
        self.lsvchoosecombo.clear()
        self.lsvchoosecombo.addItems(self.filename_list) #Update lsv combo box      
        self.lsvchoosecombo.blockSignals(False)         
        self.choose_lsv()
        
    def choose_lsv(self):
        self.lsv_chosen_name = self.lsvchoosecombo.currentText()
        self.lsv_chosen_idx = int(self.lsvchoosecombo.currentIndex())
        self.lsv_chosen_path = self.filepath_list[self.lsv_chosen_idx]
        print(self.lsv_chosen_name,self.lsv_chosen_idx)
        # Load our chosen data
        self.E = self.df_combine_E.iloc[:, self.lsv_chosen_idx].to_numpy()
        self.I = self.df_combine_I.iloc[:, self.lsv_chosen_idx].to_numpy()
        self.E = self.E[~np.isnan(self.E)] #remove nan
        self.I = self.I[~np.isnan(self.I)] #remove nan
        self.EI = np.flip(self.E/self.I)
        self.invI = np.flip(1/self.I)
        print(self.invI)
        self.plot()
        
    def plot(self):
        self.plot_EV_I.clear()
        self.curve = self.plot_EV_I.plot(self.invI,self.EI, pen=pg.mkPen('black', width=1.5))
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
    def update_xviewrange(self):
        xviewrange_slider_start = self.xviewrange.value()[0]
        xviewrange_slider_end   = self.xviewrange.value()[1]
        low_xviewrange  = self.invI[xviewrange_slider_start]
        high_xviewrange  = self.invI[xviewrange_slider_end]
        if self.EI[xviewrange_slider_start] >= np.min(self.EI[xviewrange_slider_start:xviewrange_slider_end]):
            low_yviewrange  = np.min(self.EI[xviewrange_slider_start:xviewrange_slider_end])
        else:
            low_yviewrange  = self.EI[self.xviewrange.value()[0]]
            
        if self.EI[xviewrange_slider_end] <= np.max(self.EI[xviewrange_slider_start:xviewrange_slider_end]):
            high_yviewrange  = np.max(self.EI[xviewrange_slider_start:xviewrange_slider_end])
        else:
            high_yviewrange  = self.EI[self.xviewrange.value()[1]]        
            
        # high_yrange = self.EI[self.xviewrange.value()[1]]

        self.plot_EV_I.setXRange(low_xviewrange,high_xviewrange,padding=0)
        self.plot_EV_I.setYRange(low_yviewrange,high_yviewrange,padding=0.2)
        # print(low_yrange,high_yrange)
        
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

        try: #incase start1:end1 is empty
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
            # self.datafit3.setData(self.invI[lnfit1[0]:lnfit1[-1]],y_fit2)
        except TypeError:
            pass

        # try:
        #     coeff4 = np.polyfit(self.invI[start3:end3], self.EI[start3:end3], 1)
        #     poly4 = np.poly1d(coeff4)
        #     y_fit4 = poly4(self.invI[lnfit2[0]:lnfit2[-1]])
        #     self.datafit4.setData(self.invI[lnfit2[0]:lnfit2[-1]],y_fit4)
        # except TypeError:
        #     pass 
        
        try:
            coeff3 = np.polyfit(self.invI[start3:end3], self.EI[start3:end3], 1)
            poly3 = np.poly1d(coeff3)
            y_fit3 = poly3(self.invI[lnfit2[0]:lnfit2[-1]])
            self.datafit4.setData(self.invI[lnfit2[0]:lnfit2[-1]],y_fit3)
        except TypeError:
            pass 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

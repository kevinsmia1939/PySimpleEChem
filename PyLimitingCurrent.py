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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Plotter")
        control_layout = QVBoxLayout()
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', text='E/I')
        self.plot_widget.setLabel('bottom', text='1/I')
        
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
        self.xviewrange.setFixedSize(500, 35)
        self.xviewrange.valueChanged.connect(self.update_xviewrange)
        xviewrange_layout.addWidget(self.xviewrange_text)
        xviewrange_layout.addWidget(self.xviewrange)
        control_layout.addLayout(xviewrange_layout)
        
        slider_layout = QHBoxLayout()
        self.sliderfit1_text = QLabel("Turning point fit 1:")
        self.sliderfit1 = QSlider(Qt.Horizontal)
        self.sliderfit1.setEnabled(False)
        self.sliderfit1.setFixedSize(500, 35)
        self.sliderfit1.valueChanged.connect(self.update_marker)
        slider_layout.addWidget(self.sliderfit1_text)
        slider_layout.addWidget(self.sliderfit1)
        control_layout.addLayout(slider_layout)
        
        
        
        control_layout.addStretch()

        # Main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.plot_widget, 1)
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
        print(self.filename_list)
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
        ##################################
        self.xviewrange.blockSignals(True)
        self.xviewrange.setMinimum(0)
        self.xviewrange.setMaximum(self.df_E_max_idx-1)
        self.xviewrange.setValue((0,self.df_E_max_idx-1))
        self.xviewrange.setEnabled(True)
        self.xviewrange.blockSignals(False)  
        ####################################
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
        self.plot_widget.clear()
        # Plot line

        self.curve = self.plot_widget.plot(self.invI,self.EI, pen=pg.mkPen('w', width=1.5))


        # Plot initial marker
        # self.marker = self.plot_widget.plot([self.EI[0]], [1/self.I[0]],pen=None,symbol='o',symbolBrush='r',symbolSize=10)
        
        
        # self.update_xviewrange()
        
    def update_xviewrange(self):
        xviewrange_slider_start = self.xviewrange.value()[0]
        xviewrange_slider_end   = self.xviewrange.value()[1]
        print(self.xviewrange.value()[0])
        low_xviewrange  = self.invI[xviewrange_slider_start]
        print(self.invI)
        high_xviewrange  = self.invI[xviewrange_slider_end]
        print(low_xviewrange,high_xviewrange)
        # print(high_xviewrange)
        if self.EI[xviewrange_slider_start] >= np.min(self.EI[xviewrange_slider_start:xviewrange_slider_end]):
            low_yviewrange  = np.min(self.EI[xviewrange_slider_start:xviewrange_slider_end])
        else:
            low_yviewrange  = self.EI[self.xviewrange.value()[0]]
            
        if self.EI[xviewrange_slider_end] <= np.max(self.EI[xviewrange_slider_start:xviewrange_slider_end]):
            high_yviewrange  = np.max(self.EI[xviewrange_slider_start:xviewrange_slider_end])
        else:
            high_yviewrange  = self.EI[self.xviewrange.value()[1]]        
            
        # high_yrange = self.EI[self.xviewrange.value()[1]]

        self.plot_widget.setXRange(low_xviewrange,high_xviewrange,padding=0)
        self.plot_widget.setYRange(low_yviewrange,high_yviewrange,padding=0)
        # print(low_yrange,high_yrange)
        
    def update_marker(self):
        # if self.marker is not None and 0 <= self.sliderfit1.value()[0] < len(self.x):
        sliderfit1_val = self.sliderfit1.value()
        self.marker.setData([self.E/self.I[sliderfit1_val]], [1/self.I[sliderfit1_val]])
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

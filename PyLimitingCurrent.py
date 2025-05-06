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
        open_button_layout.addWidget(self.open_button)

        control_layout.addLayout(open_button_layout)

        slider_layout = QHBoxLayout()
        self.slider_text = QLabel("Value:")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setFixedSize(500, 35)
        self.slider.valueChanged.connect(self.update_marker)
        slider_layout.addWidget(self.slider_text)
        slider_layout.addWidget(self.slider)
        control_layout.addLayout(slider_layout)
        
        


        
        xrangeview_layout = QHBoxLayout()
        self.xviewrange_text = QLabel("X-axis view")
        self.xviewrange = QRangeSlider(Qt.Horizontal)
        self.xviewrange.setEnabled(False)
        self.xviewrange.setFixedSize(500, 35)
        self.xviewrange.valueChanged.connect(self.update_xviewrange)
        xrangeview_layout.addWidget(self.xviewrange_text)
        xrangeview_layout.addWidget(self.xviewrange)
        control_layout.addLayout(xrangeview_layout)
        
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
        self.xrange_slider_res = 2000
        
    def open_file(self):
        multi_path, _ = QFileDialog.getOpenFileNames(self,
            "Open Data File",
            "",
            "Data Files (*.xlsx *.xls *.ods *.csv *.txt)"
        )
        if not multi_path:
            return
        # print(multi_path)
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
            except Exception as e:
                print(f"Error reading file: {e}")
                pass
            self.df_combine_E = pd.concat([self.df_combine_E,df["E"]],axis=1)
            self.df_combine_I = pd.concat([self.df_combine_I,df["I"]],axis=1)
            
            
        self.df_E_max = max(self.df_combine_E.max())
        self.df_E_min = min(self.df_combine_E.min())
        self.df_E_max_idx = max(self.df_combine_E.index)
        print(self.df_E_min)
        print(self.df_E_max)
        self.E_linspace = np.linspace(self.df_E_min, self.df_E_max,self.xrange_slider_res)
        self.update_xviewrange()
        self.plot()
        
    def plot(self):
        # Plot line
        self.x = self.df_combine_E.iloc[:, 0].values
        self.y = self.df_combine_I.iloc[:, 0].values
        self.plot_widget.clear()
        self.curve = self.plot_widget.plot(self.x, self.y, pen='b')

        # Plot initial marker
        self.marker = self.plot_widget.plot(
            [self.x[0]], [self.y[0]],
            pen=None,
            symbol='o',
            symbolBrush='r',
            symbolSize=10
        )

        # Configure slider
        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.x) - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)
        ##################################
        self.xviewrange.blockSignals(True)
        self.xviewrange.setMinimum(0)
        self.xviewrange.setMaximum(self.xrange_slider_res-1)
        self.xviewrange.setValue((0,self.xrange_slider_res-1))
        self.xviewrange.setEnabled(True)
        self.xviewrange.blockSignals(False)
        
    def update_xviewrange(self):
        print(self.xviewrange.value()[0])
        print(self.E_linspace)
        low_xrange  = self.E_linspace[self.xviewrange.value()[0]]
        high_xrange = self.E_linspace[self.xviewrange.value()[1]]
        print(low_xrange,high_xrange)
        self.plot_widget.setXRange(low_xrange,high_xrange,padding=0)
        
    def update_marker(self):
        # if self.marker is not None and 0 <= self.slider.value()[0] < len(self.x):
        slider_val = self.slider.value()
        self.marker.setData([self.x[slider_val]], [self.y[slider_val]])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

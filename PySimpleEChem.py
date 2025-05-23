#!/usr/bin/python3
import sys
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
# from numba import jit

# @jit

class PySimpleEChem_main(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PySimpleEChem")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget) # Use QVBoxLayout for vertical arrangement

        self.create_widgets()
        self.cv_setup_plot()

    def create_widgets(self):
        
        self.cv_concat_list_df = pd.DataFrame()
        self.cv_df_ir_currden = pd.DataFrame()
        self.cv_param_concat_df = pd.DataFrame()
        self.cv_2nd_deriv_concat_list_df = pd.DataFrame()
        self.cv_baseline_df = pd.DataFrame()
        self.cv_lines = None
        self.cv_plot_baseline_1 = None
        self.cv_plot_baseline_2 = None
        self.cv_plot_baseline_fit_1 = None
        self.cv_plot_baseline_fit_2 = None
        self.cv_plot_peak_1 = None
        self.cv_plot_peak_2 = None
        self.cv_plot_range_1 = None
        self.cv_plot_range_2 = None     
        self.cv_chosen_volt = None
        self.cv_chosen_current = None
        self.cv_peak_range = None
        self.cv_peak_pos_1_value = None
        self.cv_peak_pos_2_value = None
        self.cv_line_artist_list = []
        self.cv_plot_defl = None
        self.cv_trim_start = None
        self.cv_trim_end = None
        self.baseline_start_1 = None
        self.baseline_start_2 = None
        self.baseline_end_1 = None
        self.baseline_end_2 = None
        self.cv_pos_trim_start_box = ""
        self.cv_pos_trim_end_box = ""
        self.cv_pos_trim_start_box_val = 0
        self.cv_pos_trim_end_box_val = 0
        
        self.user_edit = True
        
        top_left_layout = QVBoxLayout()
        self.plot_tab_widget = QTabWidget()
        
        # Create the first tab for cv_plot
        tab_cv_plot = QWidget()
        tab_cv_plot_layout = QVBoxLayout(tab_cv_plot)
        self.cv_plot = pg.PlotWidget()  # Your existing plot widget
        tab_cv_plot_layout.addWidget(self.cv_plot)
        self.plot_tab_widget.addTab(tab_cv_plot, "CV Plot")
        
        # Create the second tab for cv_plot2
        tab_cv_plot2 = QWidget()
        tab_cv_plot2_layout = QVBoxLayout(tab_cv_plot2)
        self.cv_plot2 = pg.PlotWidget()  # Your new plot widget
        tab_cv_plot2_layout.addWidget(self.cv_plot2)
        self.plot_tab_widget.addTab(tab_cv_plot2, "CV Plot2")
        
        # Add the tab widget to your top left layout
        top_left_layout.addWidget(self.plot_tab_widget)      
        
        top_right_layout = QVBoxLayout() # Top right section (vertical)   
        top_right_cv_open = QHBoxLayout() #First top-right CV layout

        self.cvaddopenbutton = QPushButton("Add/Open CV file", self)
        self.cvaddopenbutton.setMenu(self.create_cv_menu())
        self.cvloadbutton = QPushButton("Load CV file", self)
        self.cvchoosecombo = QComboBox(self)
        self.cvchoosecombo.setFixedSize(300, 35)
        self.cvchoosecombo.setEditable(False)
        self.cvchoosecombo.setInsertPolicy(QComboBox.NoInsert)
        self.cvdeletebutton = QPushButton("Delete CV", self)
        top_right_cv_open.addWidget(self.cvaddopenbutton)
        top_right_cv_open.addWidget(self.cvloadbutton)
        top_right_cv_open.addWidget(self.cvchoosecombo)
        top_right_cv_open.addWidget(self.cvdeletebutton)
        top_right_layout.addLayout(top_right_cv_open)
  
        # Create a vertical layout for the top right section
        top_right_IR = QHBoxLayout()
        self.ircompenlabel = QLabel("IR compensation (ohm):")
        self.cv_ircompen_box = QLineEdit(self)
        self.cv_ircompen_box.setText("0")
        self.cv_ircompen_box.setEnabled(False)
        
        self.elec_arealabel = QLabel("Electrode area (cm<sup>2</sup>):")
        self.cv_elec_area_box = QLineEdit(self)
        self.cv_elec_area_box.setText("1")
        self.cv_elec_area_box.setEnabled(False)
        
        self.scan_rate_label = QLabel("Scan rate (V/s):")
        self.cv_scan_rate_box = QLineEdit(self)
        self.cv_scan_rate_box.setText("0")
        self.cv_scan_rate_box.setEnabled(False)
        
        top_right_IR.addWidget(self.ircompenlabel)
        top_right_IR.addWidget(self.cv_ircompen_box)
        top_right_IR.addWidget(self.elec_arealabel)
        top_right_IR.addWidget(self.cv_elec_area_box)
        top_right_IR.addWidget(self.scan_rate_label)
        top_right_IR.addWidget(self.cv_scan_rate_box)   
        top_right_layout.addLayout(top_right_IR)
               
        top_right_trim = QHBoxLayout()
        self.cv_trim_label = QLabel("Trim:")
        self.cv_trim_slider = QRangeSlider(Qt.Horizontal)
        self.cv_trim_slider.setRange(0, 1)
        self.cv_trim_slider.setValue((0, 1))
        self.cv_trim_slider.setDisabled(True)
        self.cv_pos_trim_label = QLabel("Position:")
        self.cv_pos_trim_start_box = QLineEdit(self)
        self.cv_pos_trim_start_box.setFixedSize(50, 30)
        self.cv_pos_trim_start_box.setDisabled(True)
        self.cv_pos_trim_start_box.setValidator(QIntValidator(self))
        self.cv_pos_trim_end_box = QLineEdit(self)
        self.cv_pos_trim_end_box.setFixedSize(50, 30)
        self.cv_pos_trim_end_box.setDisabled(True)
        self.cv_pos_trim_end_box.setValidator(QIntValidator(self))
        top_right_trim.addWidget(self.cv_trim_label)
        top_right_trim.addWidget(self.cv_trim_slider)
        top_right_trim.addWidget(self.cv_pos_trim_label)
        top_right_trim.addWidget(self.cv_pos_trim_start_box)
        top_right_trim.addWidget(self.cv_pos_trim_end_box)     
        top_right_layout.addLayout(top_right_trim)
        
        top_right_baseline_1 = QHBoxLayout()
        self.cv_baseline_1_label = QLabel("Baseline 1:")     
        self.cv_baseline_1_slider = QRangeSlider(Qt.Horizontal)
        self.cv_baseline_1_slider.setRange(0, 1)
        self.cv_baseline_1_slider.setValue((0, 1))
        self.cv_baseline_1_slider.setDisabled(True)
        self.cv_baseline_1_slider.barMovesAllHandles = True   
        self.cv_baseline_1_pos_label = QLabel("Position:")
        self.cv_pos_baseline_start_1_box = QLineEdit(self)
        self.cv_pos_baseline_start_1_box.setFixedSize(50, 30)
        self.cv_pos_baseline_start_1_box.setDisabled(True)
        self.cv_pos_baseline_end_1_box = QLineEdit(self)
        self.cv_pos_baseline_end_1_box.setFixedSize(50, 30)
        self.cv_pos_baseline_end_1_box.setDisabled(True)
        top_right_baseline_1.addWidget(self.cv_baseline_1_label)
        top_right_baseline_1.addWidget(self.cv_baseline_1_slider)
        top_right_baseline_1.addWidget(self.cv_baseline_1_pos_label)
        top_right_baseline_1.addWidget(self.cv_pos_baseline_start_1_box)
        top_right_baseline_1.addWidget(self.cv_pos_baseline_end_1_box)     
        top_right_layout.addLayout(top_right_baseline_1)
        
        top_right_baseline_2 = QHBoxLayout()
        self.cv_baseline_2_label = QLabel("Baseline 2:")     
        self.cv_baseline_2_slider = QRangeSlider(Qt.Horizontal)
        self.cv_baseline_2_slider.setRange(0, 1)
        self.cv_baseline_2_slider.setValue((0, 1))
        self.cv_baseline_2_slider.setDisabled(True)
        self.cv_baseline_2_slider.barMovesAllHandles = True   
        self.cv_baseline_2_pos_label = QLabel("Position:")
        self.cv_pos_baseline_start_2_box = QLineEdit(self)
        self.cv_pos_baseline_start_2_box.setFixedSize(50, 30)
        self.cv_pos_baseline_start_2_box.setDisabled(True)
        self.cv_pos_baseline_end_2_box = QLineEdit(self)
        self.cv_pos_baseline_end_2_box.setFixedSize(50, 30)
        self.cv_pos_baseline_end_2_box.setDisabled(True)
        top_right_baseline_2.addWidget(self.cv_baseline_2_label)
        top_right_baseline_2.addWidget(self.cv_baseline_2_slider)
        top_right_baseline_2.addWidget(self.cv_baseline_2_pos_label)
        top_right_baseline_2.addWidget(self.cv_pos_baseline_start_2_box)
        top_right_baseline_2.addWidget(self.cv_pos_baseline_end_2_box)     
        top_right_layout.addLayout(top_right_baseline_2)
        
        top_right_peak_search = QHBoxLayout()
        self.cv_peak_search_range_label = QLabel("Peak search range:")
        self.cv_peak_range_slider = QSlider(Qt.Horizontal)
        self.cv_peak_range_slider.setDisabled(True)
        self.cv_pos_search_range_label = QLabel("Range:")
        self.cv_pos_search_range_box = QLineEdit(self) 
        self.cv_pos_search_range_box.setFixedSize(50, 30)
        self.cv_pos_search_range_box.setDisabled(True)
        top_right_peak_search.addWidget(self.cv_peak_search_range_label)
        top_right_peak_search.addWidget(self.cv_peak_range_slider)
        top_right_peak_search.addWidget(self.cv_pos_search_range_label)
        top_right_peak_search.addWidget(self.cv_pos_search_range_box)   
        top_right_layout.addLayout(top_right_peak_search)        
        
        top_right_pos_1 = QHBoxLayout()
        self.cv_peak_pos_1_label = QLabel("Peak position 1:")
        self.cv_peak_pos_1_slider = QSlider(Qt.Horizontal)
        self.cv_peak_pos_1_slider.setDisabled(True)
        self.cv_peak_pos_mode_1_label = QLabel("Mode:")
        self.cv_peak_pos_1_combo = QComboBox(self)
        self.cv_peak_pos_1_combo.addItems(['max','min','exact','2nd derivative'])
        self.cv_peak_pos_1_combo.setCurrentText('max')
        self.cv_peak_pos_1_combo.setFixedSize(100, 35)
        self.cv_peak_pos_1_combo.setEditable(False)
        self.cv_peak_pos_1_combo.setInsertPolicy(QComboBox.NoInsert)
        self.cv_peak_pos_1_combo.setDisabled(True)
        self.cv_peak_pos_1_val_label = QLabel("Position:")
        self.cv_peak_pos_1_box = QLineEdit(self) 
        self.cv_peak_pos_1_box.setFixedSize(50, 30)
        self.cv_peak_pos_1_box.setDisabled(True)
        top_right_pos_1.addWidget(self.cv_peak_pos_1_label)
        top_right_pos_1.addWidget(self.cv_peak_pos_1_slider)
        top_right_pos_1.addWidget(self.cv_peak_pos_mode_1_label)
        top_right_pos_1.addWidget(self.cv_peak_pos_1_combo)   
        top_right_pos_1.addWidget(self.cv_peak_pos_1_val_label) 
        top_right_pos_1.addWidget(self.cv_peak_pos_1_box) 
        top_right_layout.addLayout(top_right_pos_1)        
        
        top_right_pos_2 = QHBoxLayout()
        self.cv_peak_pos_2_label = QLabel("Peak position 2:")
        self.cv_peak_pos_2_slider = QSlider(Qt.Horizontal)
        self.cv_peak_pos_2_slider.setDisabled(True)
        self.cv_peak_pos_mode_2_label = QLabel("Mode:")
        self.cv_peak_pos_2_combo = QComboBox(self)
        self.cv_peak_pos_2_combo.addItems(['max','min','exact','2nd derivative'])
        self.cv_peak_pos_2_combo.setCurrentText('min')
        self.cv_peak_pos_2_combo.setFixedSize(100, 35)
        self.cv_peak_pos_2_combo.setEditable(False)
        self.cv_peak_pos_2_combo.setInsertPolicy(QComboBox.NoInsert)
        self.cv_peak_pos_2_combo.setDisabled(True)
        self.cv_peak_pos_2_val_label = QLabel("Position:")
        self.cv_peak_pos_2_box = QLineEdit(self) 
        self.cv_peak_pos_2_box.setFixedSize(50, 30)
        self.cv_peak_pos_2_box.setDisabled(True)
        top_right_pos_2.addWidget(self.cv_peak_pos_2_label)
        top_right_pos_2.addWidget(self.cv_peak_pos_2_slider)
        top_right_pos_2.addWidget(self.cv_peak_pos_mode_2_label)
        top_right_pos_2.addWidget(self.cv_peak_pos_2_combo)   
        top_right_pos_2.addWidget(self.cv_peak_pos_2_val_label) 
        top_right_pos_2.addWidget(self.cv_peak_pos_2_box) 
        top_right_layout.addLayout(top_right_pos_2) 

        # Add Nicholson Frame
        self.nicholson_position_box = QLineEdit(self) 
        self.nicholson_position_box.setFixedSize(50, 30)
        self.cv_nicholson_frame = QFrame(self)
        self.cv_nicholson_frame.setFrameShape(QFrame.Box) # Set the frame shape to Box for a rectangular frame
        self.cv_nicholson_frame.setLineWidth(1) # Set the width of the frame lines    
        right_nicholson_frame = QVBoxLayout(self.cv_nicholson_frame)
        cv_nicholson_slider_layout = QHBoxLayout()
        self.nicholson_checkbox = QCheckBox("Nicholson method (if baseline cannot be determine)")
        self.nicholson_checkbox.setDisabled(True)
        self.switch_potential_label = QLabel("Switching potential current (Jps0)")
        self.nicholson_slider = QSlider(Qt.Horizontal, minimum=1, maximum=10, tickInterval=1)
        self.nicholson_slider.setDisabled(True)
        self.nicholson_position_label = QLabel("Position:")
        right_nicholson_frame.addWidget(self.nicholson_checkbox)     
        right_nicholson_frame.addLayout(cv_nicholson_slider_layout)              
        cv_nicholson_slider_layout.addWidget(self.switch_potential_label)
        cv_nicholson_slider_layout.addWidget(self.nicholson_slider)
        cv_nicholson_slider_layout.addWidget(self.nicholson_position_label)
        cv_nicholson_slider_layout.addWidget(self.nicholson_position_box)
        top_right_layout.addWidget(self.cv_nicholson_frame)
        


        self.cv_conc_frame = QFrame(self)
        self.cv_conc_frame.setFrameShape(QFrame.Box) # Set the frame shape to Box for a rectangular frame
        self.cv_conc_frame.setLineWidth(1) # Set the width of the frame lines  
        right_result_frame = QVBoxLayout(self.cv_conc_frame)
        cv_result_layout = QHBoxLayout()
        self.cv_bulk_conc = QLabel("Bulk concentration")
        self.cv_bulk_conc_unit = QLabel("mol/cm<sup>2</sup>")
        right_result_frame.addLayout(cv_result_layout)              
        cv_result_layout.addWidget(self.cv_bulk_conc)
        cv_result_layout.addWidget(self.cv_bulk_conc_unit)
        top_right_layout.addWidget(self.cv_conc_frame)

        #Add Help
        top_right_help = QHBoxLayout()
        self.help_label = QPushButton("Help", self)
        self.about_label = QPushButton("About", self)
        self.help_label.setFixedSize(50, 30)
        self.about_label.setFixedSize(50, 30)
        top_right_help.addWidget(self.help_label)
        top_right_help.addWidget(self.about_label)
        top_right_layout.addLayout(top_right_help)

        # Wrap top left and top right sections in QWidgets and add them to QSplitter
        top_left_widget = QWidget()
        top_left_widget.setLayout(top_left_layout)
        top_right_widget = QWidget()
        top_right_widget.setLayout(top_right_layout)
    
        # Wrap top layout in a QSplitter to make it adjustable
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(top_left_widget)
        top_splitter.addWidget(top_right_widget)
    
        # Add top splitter to main layout
        self.layout.addWidget(top_splitter)
        bottom_layout = QVBoxLayout()

        #################### Create Table #####################
        self.cv_result_table = QtWidgets.QTableView()
        # self.cv_result_display = pd.DataFrame()
        self.cv_result_display = pd.DataFrame(columns = ['file name', 'scan rate', 'Jp1', 'Jp2', 'Jp1/Jp2', 'Ep1', 'Ep2',"E\u00bd", "ΔE\u209a", "alpha1", "alpha2", "Jpc0"])
        self.cv_result_table.setModel(TableModel(self.cv_result_display))
        #######################################################
        
        bottom_layout.addWidget(self.cv_result_table)
        self.layout.addLayout(bottom_layout)

        self.cv_ircompen_box.textChanged.connect(self.cv_modify_cv_trim)
        self.cv_elec_area_box.textChanged.connect(self.cv_modify_cv_trim)
        self.cv_scan_rate_box.textChanged.connect(self.cv_modify_cv_trim)
        self.cv_trim_slider.sliderMoved.connect(self.cv_draw_all_cv)
        self.cv_pos_trim_start_box.textChanged.connect(self.cv_draw_all_cv)
        
        
        self.cvchoosecombo.textActivated.connect(self.cv_open_switch_cv)

        self.cv_baseline_1_slider.sliderMoved.connect(self.cv_draw_baseline_plot)
        self.cv_baseline_2_slider.sliderMoved.connect(self.cv_draw_baseline_plot)
        self.cv_peak_range_slider.sliderMoved.connect(self.cv_draw_marker_plot)
        self.cv_peak_pos_1_slider.sliderMoved.connect(self.cv_draw_marker_plot)
        self.cv_peak_pos_2_slider.sliderMoved.connect(self.cv_draw_marker_plot)
        
        self.cv_pos_trim_start_box.textChanged.connect(self.cv_draw_baseline_plot)
        self.cv_pos_trim_end_box.textChanged.connect(self.cv_draw_baseline_plot)
        self.cv_pos_baseline_start_1_box.textChanged.connect(self.cv_draw_baseline_plot)
        self.cv_pos_baseline_end_1_box.textChanged.connect(self.cv_draw_baseline_plot)
        self.cv_pos_baseline_start_2_box.textChanged.connect(self.cv_draw_baseline_plot)
        self.cv_pos_baseline_end_2_box.textChanged.connect(self.cv_draw_baseline_plot)
        self.cv_pos_search_range_box.textChanged.connect(self.cv_draw_marker_plot)
        self.cv_peak_pos_1_box.textChanged.connect(self.cv_draw_marker_plot)
        self.cv_peak_pos_2_box.textChanged.connect(self.cv_draw_marker_plot)

        self.cv_peak_pos_1_combo.currentIndexChanged.connect(self.cv_draw_marker_plot)
        self.cv_peak_pos_2_combo.currentIndexChanged.connect(self.cv_draw_marker_plot)
    
    def cv_setup_plot(self):
        self.cv_plot.setLabel('left', text='Current density')
        self.cv_plot.setLabel('bottom', text='Voltage')

    def create_cv_menu(self):
        cv_addload_menu = QMenu(self)
        cv_par_action = QAction('VersaSTAT(.par)', self)
        cv_par_action.triggered.connect(lambda: self.cv_open_file('.par'))
        cv_cor_action = QAction('CorrWare(.cor)', self)
        cv_cor_action.triggered.connect(lambda: self.cv_open_file('.cor'))
        cv_csv_action = QAction('Comma-separated values(.csv)', self)
        cv_csv_action.triggered.connect(lambda: self.cv_open_file('.csv'))
        cv_txt_action = QAction('Text(.txt)', self)
        cv_txt_action.triggered.connect(lambda: self.cv_open_file('.txt'))
        cv_biologic_action = QAction('Biologic(.txt)', self)
        cv_biologic_action.triggered.connect(lambda: self.cv_open_file('.txt'))
        cv_addload_menu.addAction(cv_par_action)
        cv_addload_menu.addAction(cv_cor_action)
        cv_addload_menu.addAction(cv_csv_action)
        cv_addload_menu.addAction(cv_txt_action)
        cv_addload_menu.addAction(cv_biologic_action)
        return cv_addload_menu

    def cv_open_file(self, cv_file_type):
        print("def cv_open_file enter")
        if cv_file_type == ".par":
            cv_file_format = "VersaSTAT"
        elif cv_file_type == ".cor":
            cv_file_format = "CorrWare"
        elif cv_file_type == ".csv":
            cv_file_format = "CorrWare"
        elif cv_file_type == ".txt": # TODO
            if self.sender() == self.cv_txt_action:
                cv_file_format = "text"
            elif self.sender() == self.cv_biologic_action:
                cv_file_format = "biologic"
        file_dialog = QFileDialog()
        file_dialog.setNameFilter(f'{cv_file_type} Files (*{cv_file_type})')
        cv_file_path, _ = file_dialog.getOpenFileNames(self, "Open File", "", f'{cv_file_type} Files (*{cv_file_type})')
        if cv_file_path == []:
            pass
        else:    
            for cv_file in cv_file_path:
                self.cv_df,self.cv_param_df, self.cv_2nd_deriv_concat_df = read_cv_format(cv_file,cv_file_format)  
                self.cv_df_ir_currden = pd.concat([self.cv_df_ir_currden,self.cv_df],axis=1) #Simply a copy of cv_df but will be the CV that ir compensation and current density applied
                self.cv_concat_list_df = pd.concat([self.cv_concat_list_df,self.cv_df],axis=1)
                
                self.cv_param_concat_df = pd.concat([self.cv_param_concat_df,self.cv_param_df],axis=0)
   
                self.cv_2nd_deriv_concat_list_df = pd.concat([self.cv_2nd_deriv_concat_list_df,self.cv_2nd_deriv_concat_df],axis=1)
                self.cv_result_null = pd.DataFrame([['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']],columns = ['file name', 'scan rate', 'Jp1', 'Jp2', 'Jp1/Jp2', 'Ep1', 'Ep2',"E\u00bd", "ΔE\u209a", "alpha1", "alpha2", "Jpc0"])
                
                self.cv_result_display = pd.concat([self.cv_result_display,self.cv_result_null],axis=0,ignore_index=True)
                
                self.cvchoosecombo.clear()
                self.cvchoosecombo.addItems(self.cv_param_concat_df['file name'].astype(str).tolist()) #Update cv combo box

                
            if len(self.cv_concat_list_df.columns) == 0:
                #Check if we have cv to work with
                self.empty_cv = True
            else:
                self.empty_cv = False
                self.cv_trim_slider.setDisabled(False)
                self.cv_pos_trim_start_box.setDisabled(False)
                self.cv_pos_trim_end_box.setDisabled(False)
                self.cv_baseline_1_slider.setDisabled(False)
                self.cv_pos_baseline_start_1_box.setDisabled(False)
                self.cv_pos_baseline_end_1_box.setDisabled(False)
                self.cv_baseline_2_slider.setDisabled(False)
                self.cv_pos_baseline_start_2_box.setDisabled(False)
                self.cv_pos_baseline_end_2_box.setDisabled(False) 
                self.cv_peak_range_slider.setDisabled(False)
                self.cv_pos_search_range_box.setDisabled(False)
                self.cv_peak_pos_1_slider.setDisabled(False)
                self.cv_peak_pos_1_box.setDisabled(False)
                self.cv_peak_pos_1_combo.setDisabled(False)
                self.cv_peak_pos_2_slider.setDisabled(False)
                self.cv_peak_pos_2_box.setDisabled(False)
                self.cv_peak_pos_2_combo.setDisabled(False)             
                self.nicholson_checkbox.setDisabled(False)
                
            self.cv_result_display.reset_index(drop=True, inplace=True)
            self.cv_param_concat_df.reset_index(drop=True, inplace=True)
            
            idx_file_name = 0
            for file_name in self.cv_param_concat_df['file name']:
                self.cv_result_display.loc[idx_file_name,'file name'] = file_name
                idx_file_name += 1
            
            self.cv_ircompen_box.setEnabled(True)
            self.cv_elec_area_box.setEnabled(True)
            self.cv_scan_rate_box.setEnabled(True)    

            self.cv_open_switch_cv() 
            self.cv_draw_all_cv()
        print("def cv_open_file end")
    
    def cv_open_switch_cv(self):
        print("def cv_open_switch_cv enter")
        #Activate this function when new CV is open or change
        self.cv_chosen_name = self.cvchoosecombo.currentText()
        self.cv_chosen_idx = int(self.cvchoosecombo.currentIndex())
        self.cv_chosen_path = self.cv_param_concat_df.loc[self.cv_chosen_idx]['file path']
    
        #Set initial CV volt and current
        self.cv_chosen_volt = np.array(self.cv_df_ir_currden[str(self.cv_chosen_path)+' volt'])
        self.cv_chosen_current = np.array(self.cv_df_ir_currden[str(self.cv_chosen_path)+' current'])
        self.cv_chosen_data_point_num = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['number of data points'])   


        self.cv_set_slider_val() #Now set slider to correspond to the chosen file
  
        #Set IR compensation, scan rate, and electrode area only when the user 
        #edit the value rather than copying from saved parameter
        self.user_edit = False
        self.cv_chosen_elec_area = float(self.cv_param_concat_df.loc[self.cv_chosen_idx]['elec_area'])
        self.cv_elec_area_box.setText(str(self.cv_chosen_elec_area))
        self.cv_chosen_ircompen = float(self.cv_param_concat_df.loc[self.cv_chosen_idx]['ir_compensation'])   
        self.cv_ircompen_box.setText(str(self.cv_chosen_ircompen))         
        self.cv_chosen_scan_rate = float(self.cv_param_concat_df.loc[self.cv_chosen_idx]['scan_rate'])   
        self.cv_scan_rate_box.setText(str(self.cv_chosen_scan_rate))
        self.user_edit = True
        
        # I want to fill in the values in the box and slider first before plotting
        self.cv_draw_baseline_plot()
        print("def cv_open_switch_cv end")
        
    def cv_set_slider_val(self):  
        print("def cv_set_slider_val enter")
        #When new files are loaded or switch in the combobox, set the value of the slider and boxes.
        self.cv_chosen_defl = self.cv_2nd_deriv_concat_list_df[self.cv_chosen_path]
        self.cv_trim_slider.setMaximum(self.cv_chosen_data_point_num)
        self.trim_start = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['trim_start'])
        self.trim_end = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['trim_end'])
        
        self.baseline_start_1 = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['baseline_start_1'])
        self.baseline_end_1 = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['baseline_end_1'])
        self.baseline_start_2 = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['baseline_start_2'])
        self.baseline_end_2 = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['baseline_end_2'])

        self.cv_trim_slider.setValue((self.trim_start,self.trim_end))
        self.cv_pos_trim_start_box.blockSignals(True)
        self.cv_pos_trim_start_box.setText(str(self.trim_start))
        self.cv_pos_trim_start_box.blockSignals(False)
        self.cv_pos_trim_start_box_val = self.trim_start
        self.cv_pos_trim_end_box.blockSignals(True)
        self.cv_pos_trim_end_box.setText(str(self.trim_end))
        self.cv_pos_trim_end_box.blockSignals(False)
        
        self.cv_pos_trim_end_box_val = self.trim_end
        self.cv_pos_trim_start_box.setValidator(QIntValidator(0, self.cv_chosen_data_point_num-1, self))
        self.cv_pos_trim_start_box.setText(str(self.trim_start))
        self.cv_pos_trim_end_box.setText(str(self.trim_end))
        
        self.cv_baseline_1_slider.setMaximum(self.cv_chosen_data_point_num-1)
        self.cv_baseline_2_slider.setMaximum(self.cv_chosen_data_point_num-1)       
        self.cv_peak_pos_1_slider.setMaximum(self.cv_chosen_data_point_num-1)
        self.cv_peak_pos_2_slider.setMaximum(self.cv_chosen_data_point_num-1)
        self.cv_peak_range_slider.setMaximum(self.cv_chosen_data_point_num-1)


        
        self.cv_baseline_1_slider.setValue((self.baseline_start_1,self.baseline_end_1))
        self.cv_baseline_2_slider.setValue((self.baseline_start_2,self.baseline_end_2))       
        
        self.cv_peak_range = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['peak_range'])
        self.cv_peak_range_slider.setValue(self.cv_peak_range)
        self.cv_peak_pos_1_value = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['peak_pos_1'])
        self.cv_peak_pos_1_slider.setValue(self.cv_peak_pos_1_value)
        self.cv_peak_pos_2_value = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['peak_pos_2'])
        self.cv_peak_pos_2_slider.setValue(self.cv_peak_pos_2_value)

        print("def cv_set_slider_val end")
        
    def cv_modify_cv_trim(self):
        print("def cv_modify_cv_trim enter")
        #Only calculate this value when user input them rather than programmatically from setText()
        if self.user_edit == True:
            print("def cv_modify_cv_trim user edited")
            
            #Make sure the value is a number and not negative
            try:
                self.cv_chosen_ircompen = float(self.cv_ircompen_box.text()) 
                if self.cv_chosen_ircompen < 0:
                    self.cv_chosen_ircompen = float(0)   
            except ValueError: 
                self.cv_chosen_ircompen = float(0)             
            self.cv_param_concat_df.loc[self.cv_chosen_idx,'ir_compensation'] = self.cv_chosen_ircompen
            
            try:
                self.cv_chosen_elec_area = float(self.cv_elec_area_box.text())
                if self.cv_chosen_elec_area < 0:
                    self.cv_chosen_elec_area = float(1) 
            except ValueError: 
                self.cv_chosen_elec_area = float(1)
            self.cv_param_concat_df.loc[self.cv_chosen_idx,'elec_area'] = self.cv_chosen_elec_area        
    
            try:
                self.cv_chosen_scan_rate = float(self.cv_scan_rate_box.text())
                if self.cv_chosen_scan_rate < 0:
                    self.cv_chosen_scan_rate = float(0) 
            except ValueError: 
                self.cv_chosen_scan_rate = float(0) 
            self.cv_param_concat_df.loc[self.cv_chosen_idx,'scan_rate'] = self.cv_chosen_scan_rate  
    
            #Calculate new CVs with given values
            self.cv_df_ir_currden[self.cv_chosen_path+" volt"] = self.cv_concat_list_df[self.cv_chosen_path+" volt"]+(self.cv_concat_list_df[self.cv_chosen_path+str(" current")]*self.cv_chosen_ircompen)
            self.cv_df_ir_currden[self.cv_chosen_path+" current"] = self.cv_concat_list_df[self.cv_chosen_path+" current"]/self.cv_chosen_elec_area
            
            #Save the value
            self.cv_param_concat_df.loc[self.cv_chosen_idx,'elec_area'] = self.cv_chosen_elec_area
            self.cv_param_concat_df.loc[self.cv_chosen_idx,'ir_compensation'] = self.cv_chosen_ircompen   
            self.cv_param_concat_df.loc[self.cv_chosen_idx,'scan_rate'] = self.cv_chosen_scan_rate
   
            self.cv_draw_all_cv()
        print("def cv_modify_cv_trim end")

            
    def cv_draw_all_cv(self):
        print("def cv_draw_all_cv enter")
        print(self.cv_pos_trim_start_box)
        
        #Check value of text box and update the value if it is new
        if self.cv_pos_trim_start_box.text() == '':
            self.cv_pos_trim_start_box.setText("0")
            self.cv_pos_trim_start_box_val = 0
        else:
            if int(self.cv_pos_trim_start_box.text()) < 0:
                self.cv_pos_trim_start_box.setText("0")
                self.cv_pos_trim_start_box_val = 0
            elif int(self.cv_pos_trim_start_box.text()) > self.cv_chosen_data_point_num - 1:
                self.cv_pos_trim_start_box.setText(str(self.cv_chosen_data_point_num - 1))
                self.cv_pos_trim_start_box_val = self.cv_chosen_data_point_num - 1
                
        if self.cv_trim_slider.value()[0] != self.trim_start:
            self.trim_start = self.cv_trim_slider.value()[0]
            self.cv_pos_trim_start_box.setText(str(self.trim_start))
        elif self.cv_pos_trim_start_box_val != self.trim_start:
            self.trim_start = self.cv_pos_trim_start_box_val
            self.cv_trim_slider.setValue((self.trim_start,self.trim_end))
            
        if self.cv_pos_trim_end_box.text() == '':
            self.cv_pos_trim_end_box.setText("0")
            self.cv_pos_trim_end_box_val = 0
        else:
            if int(self.cv_pos_trim_end_box.text()) < 0:
                self.cv_pos_trim_end_box.setText("0")
                self.cv_pos_trim_end_box_val = 0
            elif int(self.cv_pos_trim_end_box.text()) > self.cv_chosen_data_point_num - 1:
                self.cv_pos_trim_end_box.setText(str(self.cv_chosen_data_point_num - 1))
                self.cv_pos_trim_end_box_val = self.cv_chosen_data_point_num - 1
        if self.cv_trim_slider.value()[1] != self.trim_end:
            self.trim_end = self.cv_trim_slider.value()[1]
            self.cv_pos_trim_end_box.setText(str(self.trim_end))
        elif self.cv_pos_trim_end_box_val != self.trim_end:
            self.trim_end = self.cv_pos_trim_end_box_val
            self.cv_trim_slider.setValue((self.trim_start,self.trim_end))
        
        #Draw all CV with IR drop
        # print(self.user_edit)
        # print(self.cv_param_concat_df.to_string())
        # if self.user_edit_box == True:
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'trim_start'] = self.cv_trim_slider.value()[0]
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'trim_end'] = self.cv_trim_slider.value()[1]
        # self.user_edit_box = False
        
        
        # Remove all plot when redraw
        if self.cv_line_artist_list is not None:
            for i in self.cv_line_artist_list:
                self.cv_plot.removeItem(i)
            self.cv_line_artist_list = []   
        
        cv_select_trim_start_val = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['trim_start'])
        cv_select_trim_end_val = int(self.cv_param_concat_df.loc[self.cv_chosen_idx]['trim_end'])
        for i in range(int(self.cv_param_concat_df.shape[0])):
            cv_select_trim_start_val = self.cv_param_concat_df.loc[i]['trim_start']
            cv_select_trim_end_val = self.cv_param_concat_df.loc[i]['trim_end']       
            self.cv_lines = self.cv_plot.plot(np.array(self.cv_df_ir_currden.iloc[:,i*2])[cv_select_trim_start_val:cv_select_trim_end_val],np.array(self.cv_df_ir_currden.iloc[:,(i*2)+1])[cv_select_trim_start_val:cv_select_trim_end_val], width=1.5, pen='white')   
            self.cv_line_artist_list.append(self.cv_lines)
        max_volt = []
        min_volt = []
        max_current = []
        min_current = []
        for column in self.cv_df_ir_currden.columns:
            if column.endswith('current'):
                max_current.append(max(self.cv_df_ir_currden[column]))
                min_current.append(min(self.cv_df_ir_currden[column]))
            if column.endswith('volt'):
                max_volt.append(max(self.cv_df_ir_currden[column]))
                min_volt.append(min(self.cv_df_ir_currden[column]))
        
        self.cv_plot.setRange(xRange=[min(min_volt), max(max_volt)])
        self.cv_plot.setRange(yRange=[min(min_current), max(max_current)])
        print("def cv_draw_all_cv end")

    def cv_draw_baseline_plot(self):
        
        #Make sure that the value is correct
        if self.cv_pos_trim_start_box.text() == '':
            self.cv_pos_trim_start_box.setText("0")
            self.cv_pos_trim_start_box_val = 0
        else:
            if int(self.cv_pos_trim_start_box.text()) < 0:
                self.cv_pos_trim_start_box.setText("0")
                self.cv_pos_trim_start_box_val = 0
            elif int(self.cv_pos_trim_start_box.text()) > self.cv_chosen_data_point_num - 1:
                self.cv_pos_trim_start_box.setText(str(self.cv_chosen_data_point_num - 1))
                self.cv_pos_trim_start_box_val = self.cv_chosen_data_point_num - 1
                
        # if self.cv_baseline_1_slider.value()[0] != self.baseline_start_1:
        #     self.baseline_start_1 = self.cv_baseline_1_slider.value()[0]
        #     self.cv_pos_trim_start_box.setText(str(self.baseline_start_1))
        # elif check_val(self.cv_pos_trim_start_box,"int",self.cv_baseline_1_slider.value()[0]) != self.baseline_start_1:
        #     self.baseline_start_1 = self.cv_baseline_1_slider.value()[0]
        #     self.cv_pos_trim_start_box.setText(str(self.baseline_start_1))
        
        self.cv_plot.removeItem(self.cv_plot_baseline_1)
        self.cv_plot_baseline_1 = 0
        self.cv_plot.removeItem(self.cv_plot_baseline_2)
        self.cv_plot_baseline_2 = 1  




        # self.baseline_start_1 = self.cv_baseline_1_slider.value()[0]
        # self.baseline_end_1 = self.cv_baseline_1_slider.value()[1]
        # self.baseline_start_2 = self.cv_baseline_2_slider.value()[0]
        # self.baseline_end_2 = self.cv_baseline_2_slider.value()[1]

        volt = np.array(self.cv_df_ir_currden[str(self.cv_chosen_path)+' volt'])
        current = np.array(self.cv_df_ir_currden[str(self.cv_chosen_path)+' current'])

        self.cv_plot_baseline_1 = self.cv_plot.plot(volt[self.baseline_start_1:self.baseline_end_1],current[self.baseline_start_1:self.baseline_end_1],pen=pg.mkPen(color='red', width=4))
        self.cv_plot_baseline_2 = self.cv_plot.plot(volt[self.baseline_start_2:self.baseline_end_2],current[self.baseline_start_2:self.baseline_end_2],pen=pg.mkPen(color='skyblue', width=4))
        
        self.cv_draw_marker_plot()
        print("def cv_draw_baseline_plot end")
        
    def cv_draw_marker_plot(self):
        print("def cv_draw_marker_plot enter")
        if self.cv_df_ir_currden is not None:
            self.cv_plot.removeItem(self.cv_plot_baseline_fit_1)
            self.cv_plot_baseline_fit_1 = None
            self.cv_plot.removeItem(self.cv_plot_baseline_fit_2)
            self.cv_plot_baseline_fit_2 = None  
            self.cv_plot.removeItem(self.cv_plot_peak_1)
            self.cv_plot_peak_1 = None              
            self.cv_plot.removeItem(self.cv_plot_peak_2)
            self.cv_plot_peak_2 = None                
            self.cv_plot.removeItem(self.cv_plot_range_1)
            self.cv_plot_range_1 = None
            self.cv_plot.removeItem(self.cv_plot_range_2)
            self.cv_plot_range_2 = None   
            self.cv_plot.removeItem(self.cv_plot_defl)
            self.cv_plot_defl = None  
            
            self.cv_peak_range = self.cv_peak_range_slider.value()
            self.cv_peak_pos_1_value = self.cv_peak_pos_1_slider.value()
            self.cv_peak_pos_2_value = self.cv_peak_pos_2_slider.value()
            
            # Instead of using instance variable, I think using the local variable for voltage and current variable
            volt = np.array(self.cv_df_ir_currden[str(self.cv_chosen_path)+' volt'])
            current = np.array(self.cv_df_ir_currden[str(self.cv_chosen_path)+' current'])
            
            self.low_range_1, self.high_range_1, self.cv_peak_volt_1, self.cv_peak_curr_1, self.jp_1, self.jp_1_poly1d = get_peak_CV(self.cv_peak_pos_1_combo.currentText(), volt, current, self.cv_peak_range, self.cv_peak_pos_1_value, self.cv_baseline_1_slider.value(),self.cv_chosen_defl)
            self.low_range_2, self.high_range_2, self.cv_peak_volt_2, self.cv_peak_curr_2, self.jp_2, self.jp_2_poly1d = get_peak_CV(self.cv_peak_pos_2_combo.currentText(), volt, current, self.cv_peak_range, self.cv_peak_pos_2_value, self.cv_baseline_2_slider.value(),self.cv_chosen_defl)
            
            # Find alpha
            ep12_jp_1, jp12_jp_1, alpha_jp_1 = find_alpha(volt, current,self.cv_baseline_1_slider.value(),self.cv_peak_pos_1_value,self.jp_1_poly1d,self.jp_1,self.cv_peak_volt_1)
            ep12_jp_2, jp12_jp_2, alpha_jp_2 = find_alpha(volt, current,self.cv_baseline_2_slider.value(),self.cv_peak_pos_2_value,self.jp_2_poly1d,self.jp_2,self.cv_peak_volt_2)
            
            if self.cv_peak_pos_1_combo.currentText() == "2nd derivative" or self.cv_peak_pos_2_combo.currentText() == "2nd derivative":
                cv_chosen_defl_nonan = [int(x) for x in self.cv_chosen_defl.dropna().to_numpy()]
                self.cv_plot_defl = self.cv_plot.plot(volt[cv_chosen_defl_nonan],current[cv_chosen_defl_nonan],pen=None,symbol="o", symbolSize=8)
            
            # Flip baseline fitting depending on peak position
            baseline_fit_1 = [volt[self.baseline_start_1],volt[self.baseline_end_1]]
            print(self.baseline_start_1)
            print(baseline_fit_1)
            baseline_fit_1.sort()

            if self.cv_peak_volt_1 > baseline_fit_1[1]:
                baseline_fit_1_x1 = baseline_fit_1[0]
                baseline_fit_1_x2 = self.cv_peak_volt_1
            elif self.cv_peak_volt_1 < baseline_fit_1[0]:
                baseline_fit_1_x1 = baseline_fit_1[1]
                baseline_fit_1_x2 = self.cv_peak_volt_1
            elif self.cv_peak_volt_1 <= baseline_fit_1[1] and self.cv_peak_volt_1 >= baseline_fit_1[0]:
                baseline_fit_1_x1 = baseline_fit_1[0]
                baseline_fit_1_x2 = baseline_fit_1[1]
                
            # print(baseline_fit_1_x1)
            baseline_fit_1_y1 = self.jp_1_poly1d(baseline_fit_1_x1)
            baseline_fit_1_y2 = self.jp_1_poly1d(baseline_fit_1_x2)
    
            baseline_fit_2 = [volt[self.baseline_start_2],volt[self.baseline_end_2]]
            baseline_fit_2.sort()
            if self.cv_peak_volt_2 > baseline_fit_2[1]:
                baseline_fit_2_x1 = baseline_fit_2[0]
                baseline_fit_2_x2 = self.cv_peak_volt_2
            elif self.cv_peak_volt_2 < baseline_fit_2[0]:
                baseline_fit_2_x1 = baseline_fit_2[1]
                baseline_fit_2_x2 = self.cv_peak_volt_2
            elif self.cv_peak_volt_2 <= baseline_fit_2[1] and self.cv_peak_volt_2 >= baseline_fit_2[0]:
                baseline_fit_2_x1 = baseline_fit_2[0]
                baseline_fit_2_x2 = baseline_fit_2[1]
            baseline_fit_2_y1 = self.jp_2_poly1d(baseline_fit_2_x1)
            baseline_fit_2_y2 = self.jp_2_poly1d(baseline_fit_2_x2)        
            
            # Highlight selected data for baseline fitting
            self.cv_plot_baseline_fit_1 = self.cv_plot.plot([baseline_fit_1_x1,baseline_fit_1_x2],[baseline_fit_1_y1,baseline_fit_1_y2],pen=pg.mkPen(color='white', width=1, style=QtCore.Qt.DashLine))
            self.cv_plot_baseline_fit_2 = self.cv_plot.plot([baseline_fit_2_x1,baseline_fit_2_x2],[baseline_fit_2_y1,baseline_fit_2_y2],pen=pg.mkPen(color='white', width=1, style=QtCore.Qt.DashLine))
            # Plot peak height
            self.cv_plot_peak_1 = self.cv_plot.plot([self.cv_peak_volt_1,self.cv_peak_volt_1],[self.jp_1_poly1d(self.cv_peak_volt_1),self.cv_peak_curr_1],pen=pg.mkPen(color='white', width=1, style=QtCore.Qt.DashLine))
            self.cv_plot_peak_2 = self.cv_plot.plot([self.cv_peak_volt_2,self.cv_peak_volt_2],[self.jp_2_poly1d(self.cv_peak_volt_2),self.cv_peak_curr_2],pen=pg.mkPen(color='white', width=1, style=QtCore.Qt.DashLine))
            
            self.cv_plot_range_1 = self.cv_plot.plot([volt[int(self.low_range_1)],volt[int(self.high_range_1)]],[current[int(self.low_range_1)],current[int(self.high_range_1)]],pen=None,symbol="|")
            self.cv_plot_range_2 = self.cv_plot.plot([volt[int(self.low_range_2)],volt[int(self.high_range_2)]],[current[int(self.low_range_2)],current[int(self.high_range_2)]],pen=None,symbol="|")
            
            # Plot range that search for peak
            # Can be more efficient?????   
            self.cv_save_param()        
            
        print("def cv_draw_marker_plot end")
        
    def cv_save_param(self):   
        print("def cv_save_param enter")        
        # Write the changed parameters in cv_param_concat_df
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'trim_start'] = self.trim_start
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'trim_end'] = self.trim_end
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'baseline_start_1'] = self.baseline_start_1
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'baseline_end_1'] = self.baseline_end_1
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'baseline_start_2'] = self.baseline_start_2
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'baseline_end_2'] = self.baseline_end_2
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'peak_range'] = self.cv_peak_range
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'peak_pos_1'] = self.cv_peak_pos_1_value
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'peak_pos_2'] = self.cv_peak_pos_2_value
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'elec_area'] = self.cv_chosen_elec_area
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'ir_compensation'] = self.cv_chosen_ircompen   
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'scan_rate'] = self.cv_chosen_scan_rate
        
        #Display result
        self.cv_result_display.loc[self.cv_chosen_idx,'file name'] = self.cv_param_concat_df.loc[self.cv_chosen_idx,'file name']
        self.cv_result_display.loc[self.cv_chosen_idx,'scan rate'] = self.cv_param_concat_df.loc[self.cv_chosen_idx,'scan_rate']
        self.cv_result_display.loc[self.cv_chosen_idx,'Jp1'] = abs(self.jp_1)
        self.cv_result_display.loc[self.cv_chosen_idx,'Jp2'] = abs(self.jp_2)
        self.cv_result_display.loc[self.cv_chosen_idx,'Jp1/Jp2'] = abs(self.jp_1)/abs(self.jp_2)
        self.cv_result_display.loc[self.cv_chosen_idx,'Ep1'] = self.cv_peak_volt_1
        self.cv_result_display.loc[self.cv_chosen_idx,'Ep2'] = self.cv_peak_volt_2
        self.cv_result_display.loc[self.cv_chosen_idx,'E\u00bd'] = (self.cv_peak_volt_1+self.cv_peak_volt_2)/2
        self.cv_result_display.loc[self.cv_chosen_idx,'ΔE\u209a'] = abs(self.cv_peak_volt_1-self.cv_peak_volt_2)

        
        self.cv_result_table.setModel(TableModel(self.cv_result_display))
        print("def cv_save_param end")        
        
        
            # print(self.cv_param_concat_df.to_string())
    # def save_result(self):
    #     # Write the calculated result
    #     # print(self.cv_chosen_idx)
    #     jp1 = self.jp_1_poly1d(self.cv_peak_volt_1)-self.cv_peak_curr_1
    #     jp2 = self.jp_2_poly1d(self.cv_peak_volt_2)-self.cv_peak_curr_2   
    #     # print("jp1",jp1)
    #     # print("jp2",abs(self.jp_2_poly1d(self.cv_peak_volt_2)-self.cv_peak_curr_2))
    #     # print(self.cv_result_display)
    #     self.cv_result_table.setModel(TableModel(self.cv_result_display))


# self.low_range_1, self.high_range_1
# class helloDialog(QDialog):
#     def __init__(self, parent):
#         super().__init__(parent)
#         self.setWindowTitle("hello")
#         self.setGeometry(200, 200, 300, 150)
#         layout = QVBoxLayout()
#         label = QLabel("Say Hello", self)
#         layout.addWidget(label)
#         self.setLayout(layout)
        
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
    def rowCount(self, index):
        return self._data.shape[0]
    def columnCount(self, index):
        return self._data.shape[1]
    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = PySimpleEChem_main()
    mainWin.show()
    sys.exit(app.exec_())
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
        self.cv_plot_jsp0 = None
        self.cv_plot_jsp0_fit = None
        self.cv_plot_alpha_1 = None
        self.cv_plot_alpha_2 = None
        self.idx_jsp0 = 0
        self.jpc0 = 0.0
        self.alpha_jp_1 = 0.0
        self.alpha_jp_2 = 0.0
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
        
        # Create the second tab for Diffusion (Jp vs sqrt(scan rate))
        tab_cv_diff = QWidget()
        tab_cv_diff_layout = QVBoxLayout(tab_cv_diff)
        self.cv_plot_diff = pg.PlotWidget()
        self.cv_plot_diff.setLabel('left', text='Jp (A/cm²)')
        self.cv_plot_diff.setLabel('bottom', text='√scan rate (V/s)^½')
        self.cv_plot_diff.addLegend()
        tab_cv_diff_layout.addWidget(self.cv_plot_diff)
        self.plot_tab_widget.addTab(tab_cv_diff, "Diffusion")

        # Create the third tab for Kinetics (ln(Jp) vs Ep-E0)
        tab_cv_kin = QWidget()
        tab_cv_kin_layout = QVBoxLayout(tab_cv_kin)
        self.cv_plot_kin = pg.PlotWidget()
        self.cv_plot_kin.setLabel('left', text='ln(Jp)')
        self.cv_plot_kin.setLabel('bottom', text='Ep - E0 (V)')
        self.cv_plot_kin.addLegend()
        tab_cv_kin_layout.addWidget(self.cv_plot_kin)
        self.plot_tab_widget.addTab(tab_cv_kin, "Kinetics")
        
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
        self.cv_conc_frame.setFrameShape(QFrame.Box)
        self.cv_conc_frame.setLineWidth(1)
        right_result_frame = QVBoxLayout(self.cv_conc_frame)
        right_result_frame.addWidget(QLabel("<b>Diffusion Coefficient and Rate of Reaction</b>"))

        dk_row1 = QHBoxLayout()
        dk_row1.addWidget(QLabel("Bulk concentration (mol/cm³):"))
        self.cv_bulk_conc_box = QLineEdit("0")
        self.cv_bulk_conc_box.setDisabled(True)
        dk_row1.addWidget(self.cv_bulk_conc_box)
        dk_row1.addWidget(QLabel("No. of electrons:"))
        self.cv_elec_n_box = QLineEdit("1")
        self.cv_elec_n_box.setDisabled(True)
        dk_row1.addWidget(self.cv_elec_n_box)
        right_result_frame.addLayout(dk_row1)

        dk_row2 = QHBoxLayout()
        dk_row2.addWidget(QLabel("Avg α anodic:"))
        self.cv_alpha_ano_display = QLineEdit()
        self.cv_alpha_ano_display.setReadOnly(True)
        dk_row2.addWidget(self.cv_alpha_ano_display)
        dk_row2.addWidget(QLabel("Avg α cathodic:"))
        self.cv_alpha_cat_display = QLineEdit()
        self.cv_alpha_cat_display.setReadOnly(True)
        dk_row2.addWidget(self.cv_alpha_cat_display)
        right_result_frame.addLayout(dk_row2)

        dk_row3 = QHBoxLayout()
        dk_row3.addWidget(QLabel("D rev ano (cm²/s):"))
        self.cv_d_rev_ano = QLineEdit()
        self.cv_d_rev_ano.setReadOnly(True)
        dk_row3.addWidget(self.cv_d_rev_ano)
        dk_row3.addWidget(QLabel("D rev cat (cm²/s):"))
        self.cv_d_rev_cat = QLineEdit()
        self.cv_d_rev_cat.setReadOnly(True)
        dk_row3.addWidget(self.cv_d_rev_cat)
        right_result_frame.addLayout(dk_row3)

        dk_row4 = QHBoxLayout()
        dk_row4.addWidget(QLabel("D irr ano (cm²/s):"))
        self.cv_d_irr_ano = QLineEdit()
        self.cv_d_irr_ano.setReadOnly(True)
        dk_row4.addWidget(self.cv_d_irr_ano)
        dk_row4.addWidget(QLabel("D irr cat (cm²/s):"))
        self.cv_d_irr_cat = QLineEdit()
        self.cv_d_irr_cat.setReadOnly(True)
        dk_row4.addWidget(self.cv_d_irr_cat)
        right_result_frame.addLayout(dk_row4)

        dk_row5 = QHBoxLayout()
        dk_row5.addWidget(QLabel("k₀ anodic (cm/s):"))
        self.cv_k_ano = QLineEdit()
        self.cv_k_ano.setReadOnly(True)
        dk_row5.addWidget(self.cv_k_ano)
        dk_row5.addWidget(QLabel("k₀ cathodic (cm/s):"))
        self.cv_k_cat = QLineEdit()
        self.cv_k_cat.setReadOnly(True)
        dk_row5.addWidget(self.cv_k_cat)
        right_result_frame.addLayout(dk_row5)

        top_right_layout.addWidget(self.cv_conc_frame)

        #Add Help / About / Abbreviations
        top_right_help = QHBoxLayout()
        self.help_button = QPushButton("Help", self)
        self.about_button = QPushButton("About", self)
        self.abbrev_button = QPushButton("What is this?", self)
        top_right_help.addWidget(self.help_button)
        top_right_help.addWidget(self.about_button)
        top_right_help.addWidget(self.abbrev_button)
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
        self.cv_pos_trim_end_box.textChanged.connect(self.cv_draw_all_cv)

        self.cvchoosecombo.textActivated.connect(self.cv_open_switch_cv)

        self.cv_baseline_1_slider.sliderMoved.connect(self.cv_draw_baseline_plot)
        self.cv_baseline_2_slider.sliderMoved.connect(self.cv_draw_baseline_plot)
        self.cv_baseline_1_slider.sliderReleased.connect(self.cv_draw_marker_plot)
        self.cv_baseline_2_slider.sliderReleased.connect(self.cv_draw_marker_plot)
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

        self.nicholson_checkbox.stateChanged.connect(self.cv_draw_marker_plot)
        self.nicholson_slider.sliderMoved.connect(self.cv_draw_marker_plot)
        self.nicholson_position_box.textChanged.connect(self.cv_draw_marker_plot)

        self.cv_bulk_conc_box.textChanged.connect(self.cv_calc_diffusion_kinetics)
        self.cv_elec_n_box.textChanged.connect(self.cv_calc_diffusion_kinetics)

        self.help_button.clicked.connect(self.show_help)
        self.about_button.clicked.connect(self.show_about)
        self.abbrev_button.clicked.connect(self.show_abbreviations)
        self.cvdeletebutton.clicked.connect(self.cv_delete_cv)

    def cv_setup_plot(self):
        self.cv_plot.setLabel('left', text='Current density')
        self.cv_plot.setLabel('bottom', text='Voltage')
        # Pre-allocate baseline highlight lines so setData() can be used during drag
        # (avoids removeItem + plot overhead on every slider event)
        self.cv_plot_baseline_1 = self.cv_plot.plot([], [], pen=pg.mkPen(color='red', width=4))
        self.cv_plot_baseline_2 = self.cv_plot.plot([], [], pen=pg.mkPen(color='skyblue', width=4))

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

    def cv_delete_cv(self):
        if self.empty_cv or self.cvchoosecombo.count() == 0:
            return

        del_idx = int(self.cvchoosecombo.currentIndex())
        del_path = self.cv_param_concat_df.loc[del_idx]['file path']

        # Remove volt/current columns from CV data frames
        volt_col = str(del_path) + ' volt'
        curr_col = str(del_path) + ' current'
        for col in [volt_col, curr_col]:
            if col in self.cv_concat_list_df.columns:
                self.cv_concat_list_df.drop(columns=[col], inplace=True)
            if col in self.cv_df_ir_currden.columns:
                self.cv_df_ir_currden.drop(columns=[col], inplace=True)

        # Remove 2nd-derivative column
        if str(del_path) in self.cv_2nd_deriv_concat_list_df.columns:
            self.cv_2nd_deriv_concat_list_df.drop(columns=[str(del_path)], inplace=True)

        # Remove param row and result row, reset indices
        self.cv_param_concat_df.drop(index=del_idx, inplace=True)
        self.cv_param_concat_df.reset_index(drop=True, inplace=True)
        self.cv_result_display.drop(index=del_idx, inplace=True)
        self.cv_result_display.reset_index(drop=True, inplace=True)

        # Update combo box
        self.cvchoosecombo.clear()
        if len(self.cv_param_concat_df) == 0:
            self.empty_cv = True
            # Disable all controls
            self.cv_trim_slider.setDisabled(True)
            self.cv_pos_trim_start_box.setDisabled(True)
            self.cv_pos_trim_end_box.setDisabled(True)
            self.cv_baseline_1_slider.setDisabled(True)
            self.cv_pos_baseline_start_1_box.setDisabled(True)
            self.cv_pos_baseline_end_1_box.setDisabled(True)
            self.cv_baseline_2_slider.setDisabled(True)
            self.cv_pos_baseline_start_2_box.setDisabled(True)
            self.cv_pos_baseline_end_2_box.setDisabled(True)
            self.cv_peak_range_slider.setDisabled(True)
            self.cv_pos_search_range_box.setDisabled(True)
            self.cv_peak_pos_1_slider.setDisabled(True)
            self.cv_peak_pos_1_box.setDisabled(True)
            self.cv_peak_pos_1_combo.setDisabled(True)
            self.cv_peak_pos_2_slider.setDisabled(True)
            self.cv_peak_pos_2_box.setDisabled(True)
            self.cv_peak_pos_2_combo.setDisabled(True)
            self.nicholson_checkbox.setDisabled(True)
            self.nicholson_slider.setDisabled(True)
            self.nicholson_position_box.setDisabled(True)
            self.cv_bulk_conc_box.setDisabled(True)
            self.cv_elec_n_box.setDisabled(True)
            self.cv_ircompen_box.setEnabled(False)
            self.cv_elec_area_box.setEnabled(False)
            self.cv_scan_rate_box.setEnabled(False)
            # Clear all plots — use setData([], []) on pre-allocated baseline items
            # so they stay valid; remove dynamic CV lines manually
            for item in list(self.cv_line_artist_list):
                self.cv_plot.removeItem(item)
            self.cv_line_artist_list = []
            self.cv_lines = None
            self.cv_plot_baseline_1.setData([], [])
            self.cv_plot_baseline_2.setData([], [])
            self.cv_plot_diff.clear()
            self.cv_plot_kin.clear()
            # Clear marker overlay items
            for attr in ('cv_plot_baseline_fit_1', 'cv_plot_baseline_fit_2',
                         'cv_plot_peak_1', 'cv_plot_peak_2',
                         'cv_plot_range_1', 'cv_plot_range_2',
                         'cv_plot_defl', 'cv_plot_jsp0', 'cv_plot_jsp0_fit',
                         'cv_plot_alpha_1', 'cv_plot_alpha_2'):
                item = getattr(self, attr)
                if item is not None:
                    self.cv_plot.removeItem(item)
                    setattr(self, attr, None)
            # Update result table
            self.cv_result_table.setModel(TableModel(self.cv_result_display))
        else:
            self.cvchoosecombo.addItems(self.cv_param_concat_df['file name'].astype(str).tolist())
            new_idx = min(del_idx, len(self.cv_param_concat_df) - 1)
            self.cvchoosecombo.setCurrentIndex(new_idx)
            self.cv_open_switch_cv()
            self.cv_draw_all_cv()

    def cv_open_file(self, cv_file_type):
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
                self.nicholson_slider.setDisabled(False)
                self.nicholson_position_box.setDisabled(False)
                self.cv_bulk_conc_box.setDisabled(False)
                self.cv_elec_n_box.setDisabled(False)
                
            self.cv_result_display.reset_index(drop=True, inplace=True)
            self.cv_param_concat_df.reset_index(drop=True, inplace=True)
            
            idx_file_name = 0
            for file_name in self.cv_param_concat_df['file name']:
                self.cv_result_display.loc[idx_file_name,'file name'] = file_name
                idx_file_name += 1
            
            self.cv_ircompen_box.setEnabled(True)
            self.cv_elec_area_box.setEnabled(True)
            self.cv_scan_rate_box.setEnabled(True)

            # Switch the combo box to the last added file so it becomes active
            last_idx = len(self.cv_param_concat_df) - 1
            self.cvchoosecombo.setCurrentIndex(last_idx)

            self.cv_open_switch_cv()
            self.cv_draw_all_cv()
    
    def cv_open_switch_cv(self):
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
        nicholson_bool_val = bool(self.cv_param_concat_df.loc[self.cv_chosen_idx]['nicholson_bool'])
        self.nicholson_checkbox.blockSignals(True)
        self.nicholson_checkbox.setChecked(nicholson_bool_val)
        self.nicholson_checkbox.blockSignals(False)
        self.user_edit = True
        
        # I want to fill in the values in the box and slider first before plotting
        self.cv_draw_baseline_plot()
        
    def cv_set_slider_val(self):  
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

        self.nicholson_slider.setMaximum(self.cv_chosen_data_point_num - 1)
        self.idx_jsp0 = int(float(self.cv_param_concat_df.loc[self.cv_chosen_idx]['jsp0']))
        self.nicholson_slider.setValue(self.idx_jsp0)
        self.nicholson_position_box.blockSignals(True)
        self.nicholson_position_box.setText(str(self.idx_jsp0))
        self.nicholson_position_box.blockSignals(False)
        
    def cv_modify_cv_trim(self):
        #Only calculate this value when user input them rather than programmatically from setText()
        if self.user_edit == True:
            
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
            # Refresh cached numpy arrays so subsequent draw calls see the new values
            self.cv_chosen_volt = np.array(self.cv_df_ir_currden[str(self.cv_chosen_path)+' volt'])
            self.cv_chosen_current = np.array(self.cv_df_ir_currden[str(self.cv_chosen_path)+' current'])
            
            #Save the value
            self.cv_param_concat_df.loc[self.cv_chosen_idx,'elec_area'] = self.cv_chosen_elec_area
            self.cv_param_concat_df.loc[self.cv_chosen_idx,'ir_compensation'] = self.cv_chosen_ircompen   
            self.cv_param_concat_df.loc[self.cv_chosen_idx,'scan_rate'] = self.cv_chosen_scan_rate
   
            self.cv_draw_all_cv()

            
    def cv_draw_all_cv(self):
        max_idx = self.cv_chosen_data_point_num - 1

        def _clamp_int(text, lo, hi, fallback):
            try:
                v = int(text)
                return max(lo, min(hi, v))
            except (ValueError, TypeError):
                return fallback

        sender = self.sender()
        if sender == self.cv_trim_slider:
            # Slider moved: read slider, push to text boxes without re-triggering
            self.trim_start = self.cv_trim_slider.value()[0]
            self.trim_end   = self.cv_trim_slider.value()[1]
            self.cv_pos_trim_start_box.blockSignals(True)
            self.cv_pos_trim_start_box.setText(str(self.trim_start))
            self.cv_pos_trim_start_box.blockSignals(False)
            self.cv_pos_trim_end_box.blockSignals(True)
            self.cv_pos_trim_end_box.setText(str(self.trim_end))
            self.cv_pos_trim_end_box.blockSignals(False)
        else:
            # Text box changed: read text boxes, push to slider without re-triggering
            self.trim_start = _clamp_int(self.cv_pos_trim_start_box.text(), 0, max_idx, self.trim_start)
            self.trim_end   = _clamp_int(self.cv_pos_trim_end_box.text(),   0, max_idx, self.trim_end)
            self.cv_trim_slider.blockSignals(True)
            self.cv_trim_slider.setValue((self.trim_start, self.trim_end))
            self.cv_trim_slider.blockSignals(False)

        self.cv_param_concat_df.loc[self.cv_chosen_idx, 'trim_start'] = self.trim_start
        self.cv_param_concat_df.loc[self.cv_chosen_idx, 'trim_end']   = self.trim_end

        # Remove all CV lines before redrawing
        if self.cv_line_artist_list is not None:
            for i in self.cv_line_artist_list:
                self.cv_plot.removeItem(i)
            self.cv_line_artist_list = []

        for i in range(int(self.cv_param_concat_df.shape[0])):
            cv_select_trim_start_val = int(self.cv_param_concat_df.loc[i]['trim_start'])
            cv_select_trim_end_val   = int(self.cv_param_concat_df.loc[i]['trim_end'])
            self.cv_lines = self.cv_plot.plot(
                np.array(self.cv_df_ir_currden.iloc[:, i*2])[cv_select_trim_start_val:cv_select_trim_end_val],
                np.array(self.cv_df_ir_currden.iloc[:, (i*2)+1])[cv_select_trim_start_val:cv_select_trim_end_val],
                width=1.5, pen='white')
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

        # Determine which widget triggered this call and sync slider <-> text box
        sender = self.sender()
        max_idx = self.cv_chosen_data_point_num - 1
        if sender in (self.cv_baseline_1_slider, self.cv_baseline_2_slider):
            # Slider moved: read slider and update text boxes
            self.baseline_start_1 = self.cv_baseline_1_slider.value()[0]
            self.baseline_end_1 = self.cv_baseline_1_slider.value()[1]
            self.baseline_start_2 = self.cv_baseline_2_slider.value()[0]
            self.baseline_end_2 = self.cv_baseline_2_slider.value()[1]
            for box, val in ((self.cv_pos_baseline_start_1_box, self.baseline_start_1),
                             (self.cv_pos_baseline_end_1_box,   self.baseline_end_1),
                             (self.cv_pos_baseline_start_2_box, self.baseline_start_2),
                             (self.cv_pos_baseline_end_2_box,   self.baseline_end_2)):
                box.blockSignals(True)
                box.setText(str(val))
                box.blockSignals(False)
        else:
            # Text box changed (or initial load): read text boxes and update sliders
            def _clamp_int(text, lo, hi, fallback):
                try:
                    v = int(text)
                    return max(lo, min(hi, v))
                except (ValueError, TypeError):
                    return fallback
            self.baseline_start_1 = _clamp_int(self.cv_pos_baseline_start_1_box.text(), 0, max_idx, self.baseline_start_1)
            self.baseline_end_1   = _clamp_int(self.cv_pos_baseline_end_1_box.text(),   0, max_idx, self.baseline_end_1)
            self.baseline_start_2 = _clamp_int(self.cv_pos_baseline_start_2_box.text(), 0, max_idx, self.baseline_start_2)
            self.baseline_end_2   = _clamp_int(self.cv_pos_baseline_end_2_box.text(),   0, max_idx, self.baseline_end_2)
            self.cv_baseline_1_slider.blockSignals(True)
            self.cv_baseline_1_slider.setValue((self.baseline_start_1, self.baseline_end_1))
            self.cv_baseline_1_slider.blockSignals(False)
            self.cv_baseline_2_slider.blockSignals(True)
            self.cv_baseline_2_slider.setValue((self.baseline_start_2, self.baseline_end_2))
            self.cv_baseline_2_slider.blockSignals(False)

        volt = self.cv_chosen_volt
        current = self.cv_chosen_current

        # Use setData() on pre-allocated items — no removeItem/plot overhead
        self.cv_plot_baseline_1.setData(volt[self.baseline_start_1:self.baseline_end_1],
                                        current[self.baseline_start_1:self.baseline_end_1])
        self.cv_plot_baseline_2.setData(volt[self.baseline_start_2:self.baseline_end_2],
                                        current[self.baseline_start_2:self.baseline_end_2])

        # During slider drag, only update the highlight — defer the heavy marker
        # computation to sliderReleased (connected separately)
        if sender not in (self.cv_baseline_1_slider, self.cv_baseline_2_slider):
            self.cv_draw_marker_plot()
        
    def cv_draw_marker_plot(self):
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
            self.cv_plot.removeItem(self.cv_plot_jsp0)
            self.cv_plot_jsp0 = None
            self.cv_plot.removeItem(self.cv_plot_jsp0_fit)
            self.cv_plot_jsp0_fit = None
            self.cv_plot.removeItem(self.cv_plot_alpha_1)
            self.cv_plot_alpha_1 = None
            self.cv_plot.removeItem(self.cv_plot_alpha_2)
            self.cv_plot_alpha_2 = None

            # Sync sliders <-> text boxes for peak controls and Nicholson slider
            max_idx = self.cv_chosen_data_point_num - 1
            def _clamp_int(text, lo, hi, fallback):
                try:
                    v = int(text)
                    return max(lo, min(hi, v))
                except (ValueError, TypeError):
                    return fallback

            sender = self.sender()
            if sender in (self.cv_peak_range_slider, self.cv_peak_pos_1_slider, self.cv_peak_pos_2_slider):
                self.cv_peak_range = self.cv_peak_range_slider.value()
                self.cv_peak_pos_1_value = self.cv_peak_pos_1_slider.value()
                self.cv_peak_pos_2_value = self.cv_peak_pos_2_slider.value()
                for box, val in ((self.cv_pos_search_range_box, self.cv_peak_range),
                                 (self.cv_peak_pos_1_box,       self.cv_peak_pos_1_value),
                                 (self.cv_peak_pos_2_box,       self.cv_peak_pos_2_value)):
                    box.blockSignals(True)
                    box.setText(str(val))
                    box.blockSignals(False)
            elif sender in (self.cv_pos_search_range_box, self.cv_peak_pos_1_box, self.cv_peak_pos_2_box):
                self.cv_peak_range = _clamp_int(self.cv_pos_search_range_box.text(), 0, max_idx, self.cv_peak_range)
                self.cv_peak_pos_1_value = _clamp_int(self.cv_peak_pos_1_box.text(), 0, max_idx, self.cv_peak_pos_1_value)
                self.cv_peak_pos_2_value = _clamp_int(self.cv_peak_pos_2_box.text(), 0, max_idx, self.cv_peak_pos_2_value)
                self.cv_peak_range_slider.blockSignals(True)
                self.cv_peak_range_slider.setValue(self.cv_peak_range)
                self.cv_peak_range_slider.blockSignals(False)
                self.cv_peak_pos_1_slider.blockSignals(True)
                self.cv_peak_pos_1_slider.setValue(self.cv_peak_pos_1_value)
                self.cv_peak_pos_1_slider.blockSignals(False)
                self.cv_peak_pos_2_slider.blockSignals(True)
                self.cv_peak_pos_2_slider.setValue(self.cv_peak_pos_2_value)
                self.cv_peak_pos_2_slider.blockSignals(False)
            elif sender == self.nicholson_slider:
                self.idx_jsp0 = self.nicholson_slider.value()
                self.nicholson_position_box.blockSignals(True)
                self.nicholson_position_box.setText(str(self.idx_jsp0))
                self.nicholson_position_box.blockSignals(False)
                self.cv_peak_range = self.cv_peak_range_slider.value()
                self.cv_peak_pos_1_value = self.cv_peak_pos_1_slider.value()
                self.cv_peak_pos_2_value = self.cv_peak_pos_2_slider.value()
            elif sender == self.nicholson_position_box:
                self.idx_jsp0 = _clamp_int(self.nicholson_position_box.text(), 0, max_idx, self.idx_jsp0)
                self.nicholson_slider.blockSignals(True)
                self.nicholson_slider.setValue(self.idx_jsp0)
                self.nicholson_slider.blockSignals(False)
                self.cv_peak_range = self.cv_peak_range_slider.value()
                self.cv_peak_pos_1_value = self.cv_peak_pos_1_slider.value()
                self.cv_peak_pos_2_value = self.cv_peak_pos_2_slider.value()
            else:
                # Combo box, checkbox, or programmatic call: read from all sliders
                self.cv_peak_range = self.cv_peak_range_slider.value()
                self.cv_peak_pos_1_value = self.cv_peak_pos_1_slider.value()
                self.cv_peak_pos_2_value = self.cv_peak_pos_2_slider.value()
                self.idx_jsp0 = self.nicholson_slider.value()

            volt = self.cv_chosen_volt
            current = self.cv_chosen_current

            nicholson_bool = self.nicholson_checkbox.isChecked()

            # Peak 1 always uses baseline 1
            self.low_range_1, self.high_range_1, self.cv_peak_volt_1, self.cv_peak_curr_1, self.jp_1, self.jp_1_poly1d = get_peak_CV(
                self.cv_peak_pos_1_combo.currentText(), volt, current, self.cv_peak_range,
                self.cv_peak_pos_1_value, self.cv_baseline_1_slider.value(), self.cv_chosen_defl)
            ep12_jp_1, jp12_jp_1, self.alpha_jp_1 = find_alpha(
                volt, current, self.cv_baseline_1_slider.value(),
                self.cv_peak_pos_1_value, self.jp_1_poly1d, self.jp_1, self.cv_peak_volt_1)

            if not nicholson_bool:
                # Standard method: peak 2 uses baseline 2
                self.low_range_2, self.high_range_2, self.cv_peak_volt_2, self.cv_peak_curr_2, self.jp_2, self.jp_2_poly1d = get_peak_CV(
                    self.cv_peak_pos_2_combo.currentText(), volt, current, self.cv_peak_range,
                    self.cv_peak_pos_2_value, self.cv_baseline_2_slider.value(), self.cv_chosen_defl)
                ep12_jp_2, jp12_jp_2, self.alpha_jp_2 = find_alpha(
                    volt, current, self.cv_baseline_2_slider.value(),
                    self.cv_peak_pos_2_value, self.jp_2_poly1d, self.jp_2, self.cv_peak_volt_2)
                self.jpc0 = 0.0
                # Plot half-peak alpha markers for both peaks
                self.cv_plot_alpha_1 = self.cv_plot.plot(
                    [ep12_jp_1], [jp12_jp_1], pen=None, symbol='x',
                    symbolSize=10, symbolPen=pg.mkPen(color='orange', width=2))
                self.cv_plot_alpha_2 = self.cv_plot.plot(
                    [ep12_jp_2], [jp12_jp_2], pen=None, symbol='x',
                    symbolSize=10, symbolPen=pg.mkPen(color='orange', width=2))
            else:
                # Nicholson method (Nicholson, R. S. Anal. Chem. 1966, 38, 1406):
                # jpc0 = cathodic peak measured against the anodic (baseline 1) fit
                self.low_range_2, self.high_range_2, self.cv_peak_volt_2, self.cv_peak_curr_2, jpc0_raw, _ = get_peak_CV(
                    self.cv_peak_pos_2_combo.currentText(), volt, current, self.cv_peak_range,
                    self.cv_peak_pos_2_value, self.cv_baseline_1_slider.value(), self.cv_chosen_defl)
                # Re-use anodic poly for peak-2 visualisation
                self.jp_2_poly1d = self.jp_1_poly1d
                # jsp0 = current at switching potential relative to the anodic baseline
                jsp0_volt = volt[self.idx_jsp0]
                jsp0 = current[self.idx_jsp0] - self.jp_1_poly1d(jsp0_volt)
                jpa  = np.abs(self.jp_1)
                self.jpc0 = np.abs(jpc0_raw)
                jsp0_abs  = np.abs(jsp0)
                # Nicholson formula
                self.jp_2 = np.abs(jpa * ((self.jpc0 / jpa) + (0.485 * jsp0_abs / jpa) + 0.086))
                # Alpha only for peak 1 in Nicholson mode; cathodic is estimated
                self.alpha_jp_2 = np.nan
                self.cv_plot_alpha_1 = self.cv_plot.plot(
                    [ep12_jp_1], [jp12_jp_1], pen=None, symbol='x',
                    symbolSize=10, symbolPen=pg.mkPen(color='orange', width=2))
                # Draw jsp0 indicator: dashed baseline extension + vertical line
                self.cv_plot_jsp0_fit = self.cv_plot.plot(
                    [volt[self.baseline_start_1], jsp0_volt],
                    [self.jp_1_poly1d(volt[self.baseline_start_1]), self.jp_1_poly1d(jsp0_volt)],
                    pen=pg.mkPen(color='green', width=1, style=QtCore.Qt.DashLine))
                self.cv_plot_jsp0 = self.cv_plot.plot(
                    [jsp0_volt, jsp0_volt],
                    [self.jp_1_poly1d(jsp0_volt), current[self.idx_jsp0]],
                    pen=pg.mkPen(color='cyan', width=2))

            if self.cv_peak_pos_1_combo.currentText() == "2nd derivative" or self.cv_peak_pos_2_combo.currentText() == "2nd derivative":
                cv_chosen_defl_nonan = [int(x) for x in self.cv_chosen_defl.dropna().to_numpy()]
                self.cv_plot_defl = self.cv_plot.plot(
                    volt[cv_chosen_defl_nonan], current[cv_chosen_defl_nonan],
                    pen=None, symbol="o", symbolSize=8)

            # Baseline fit extrapolation lines
            def _baseline_fit_endpoints(peak_volt, b_x0, b_x1, poly1d):
                pts = sorted([b_x0, b_x1])
                if peak_volt > pts[1]:
                    x1, x2 = pts[0], peak_volt
                elif peak_volt < pts[0]:
                    x1, x2 = pts[1], peak_volt
                else:
                    x1, x2 = pts[0], pts[1]
                return x1, x2, poly1d(x1), poly1d(x2)

            b1_x1, b1_x2, b1_y1, b1_y2 = _baseline_fit_endpoints(
                self.cv_peak_volt_1,
                volt[self.baseline_start_1], volt[self.baseline_end_1],
                self.jp_1_poly1d)

            if not nicholson_bool:
                b2_x1, b2_x2, b2_y1, b2_y2 = _baseline_fit_endpoints(
                    self.cv_peak_volt_2,
                    volt[self.baseline_start_2], volt[self.baseline_end_2],
                    self.jp_2_poly1d)
            else:
                # Show anodic baseline extended to peak 2 voltage
                b2_x1, b2_x2, b2_y1, b2_y2 = _baseline_fit_endpoints(
                    self.cv_peak_volt_2,
                    volt[self.baseline_start_1], volt[self.baseline_end_1],
                    self.jp_1_poly1d)

            self.cv_plot_baseline_fit_1 = self.cv_plot.plot(
                [b1_x1, b1_x2], [b1_y1, b1_y2],
                pen=pg.mkPen(color='white', width=1, style=QtCore.Qt.DashLine))
            self.cv_plot_baseline_fit_2 = self.cv_plot.plot(
                [b2_x1, b2_x2], [b2_y1, b2_y2],
                pen=pg.mkPen(color='white', width=1, style=QtCore.Qt.DashLine))
            self.cv_plot_peak_1 = self.cv_plot.plot(
                [self.cv_peak_volt_1, self.cv_peak_volt_1],
                [self.jp_1_poly1d(self.cv_peak_volt_1), self.cv_peak_curr_1],
                pen=pg.mkPen(color='white', width=1, style=QtCore.Qt.DashLine))
            self.cv_plot_peak_2 = self.cv_plot.plot(
                [self.cv_peak_volt_2, self.cv_peak_volt_2],
                [self.jp_2_poly1d(self.cv_peak_volt_2), self.cv_peak_curr_2],
                pen=pg.mkPen(color='white', width=1, style=QtCore.Qt.DashLine))
            self.cv_plot_range_1 = self.cv_plot.plot(
                [volt[int(self.low_range_1)], volt[int(self.high_range_1)]],
                [current[int(self.low_range_1)], current[int(self.high_range_1)]],
                pen=None, symbol="|")
            self.cv_plot_range_2 = self.cv_plot.plot(
                [volt[int(self.low_range_2)], volt[int(self.high_range_2)]],
                [current[int(self.low_range_2)], current[int(self.high_range_2)]],
                pen=None, symbol="|")

            self.cv_save_param()
        
    def cv_save_param(self):         
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
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'nicholson_bool'] = self.nicholson_checkbox.isChecked()
        self.cv_param_concat_df.loc[self.cv_chosen_idx,'jsp0'] = self.idx_jsp0

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
        self.cv_result_display.loc[self.cv_chosen_idx,'alpha1'] = self.alpha_jp_1
        self.cv_result_display.loc[self.cv_chosen_idx,'alpha2'] = self.alpha_jp_2
        self.cv_result_display.loc[self.cv_chosen_idx,'Jpc0'] = self.jpc0

        
        self.cv_result_table.setModel(TableModel(self.cv_result_display))
        self.cv_calc_diffusion_kinetics()

    def cv_calc_diffusion_kinetics(self):
        """Compute diffusion coefficients and rate constant from all loaded CVs."""
        try:
            bulk_conc = float(self.cv_bulk_conc_box.text())
        except ValueError:
            bulk_conc = 0.0
        try:
            elec_n = float(self.cv_elec_n_box.text())
            if elec_n <= 0:
                elec_n = 1.0
        except ValueError:
            elec_n = 1.0

        # Collect numeric rows — skip rows that still have placeholder '-'
        try:
            df = self.cv_result_display.copy()
            df = df[pd.to_numeric(df['scan rate'], errors='coerce').notna()]
            df = df[pd.to_numeric(df['Jp1'],      errors='coerce').notna()]
            df = df[pd.to_numeric(df['Jp2'],      errors='coerce').notna()]
            if len(df) < 2:
                return  # Need at least 2 scan rates for a meaningful fit
            scan    = df['scan rate'].astype(float).to_numpy()
            jpa_arr = df['Jp1'].astype(float).to_numpy()
            jpc_arr = df['Jp2'].astype(float).to_numpy()
            alpha_ano_arr = pd.to_numeric(df['alpha1'], errors='coerce').dropna().to_numpy()
            alpha_cat_arr = pd.to_numeric(df['alpha2'], errors='coerce').dropna().to_numpy()
            delta_e_arr   = pd.to_numeric(df['ΔE\u209a'], errors='coerce').dropna().to_numpy()
        except Exception:
            return

        alpha_ano = float(np.nanmean(alpha_ano_arr)) if len(alpha_ano_arr) else 0.5
        alpha_cat = float(np.nanmean(alpha_cat_arr)) if len(alpha_cat_arr) else 0.5
        self.cv_alpha_ano_display.setText(f"{alpha_ano:.5f}")
        self.cv_alpha_cat_display.setText(f"{alpha_cat:.5f}")

        # ── Diffusion plot ──────────────────────────────────────────────────────
        self.cv_plot_diff.clear()
        if bulk_conc > 0:
            try:
                sqrt_scan_ano, jpa_fit, D_irr_a, D_rev_a, r2_ano = diffusion(scan, jpa_arr, alpha_ano, bulk_conc, elec_n)
                sqrt_scan_cat, jpc_fit, D_irr_c, D_rev_c, r2_cat = diffusion(scan, jpc_arr, alpha_cat, bulk_conc, elec_n)
                self.cv_plot_diff.plot(sqrt_scan_ano, jpa_fit, pen=pg.mkPen('r', width=2), name='Jp1 fit')
                self.cv_plot_diff.plot(sqrt_scan_cat, jpc_fit, pen=pg.mkPen('b', width=2), name='Jp2 fit')
                self.cv_d_rev_ano.setText(f"{D_rev_a:.4e}")
                self.cv_d_rev_cat.setText(f"{D_rev_c:.4e}")
                self.cv_d_irr_ano.setText(f"{D_irr_a:.4e}")
                self.cv_d_irr_cat.setText(f"{D_irr_c:.4e}")
            except Exception:
                pass
        sqrt_scan = np.sqrt(scan)
        self.cv_plot_diff.plot(sqrt_scan, jpa_arr, pen=None, symbol='o',
                               symbolBrush='r', symbolSize=8, name='Jp1')
        self.cv_plot_diff.plot(sqrt_scan, jpc_arr, pen=None, symbol='o',
                               symbolBrush='b', symbolSize=8, name='Jp2')

        # ── Kinetics plot ───────────────────────────────────────────────────────
        self.cv_plot_kin.clear()
        if len(delta_e_arr) >= 2 and bulk_conc > 0:
            e_e0_arr = delta_e_arr / 2.0
            try:
                lnjpa, lnjpa_fit, k0_a, _, _, _ = reaction_rate(e_e0_arr, jpa_arr[:len(e_e0_arr)], bulk_conc, elec_n)
                lnjpc, lnjpc_fit, k0_c, _, _, _ = reaction_rate(e_e0_arr, jpc_arr[:len(e_e0_arr)], bulk_conc, elec_n)
                self.cv_plot_kin.plot(e_e0_arr, lnjpa_fit, pen=pg.mkPen('r', width=2), name='Jp1 fit')
                self.cv_plot_kin.plot(e_e0_arr, lnjpc_fit, pen=pg.mkPen('b', width=2), name='Jp2 fit')
                self.cv_k_ano.setText(f"{k0_a:.4e}")
                self.cv_k_cat.setText(f"{k0_c:.4e}")
            except Exception:
                pass
            lnjpa_all = np.log(np.abs(jpa_arr[:len(e_e0_arr)]))
            lnjpc_all = np.log(np.abs(jpc_arr[:len(e_e0_arr)]))
            self.cv_plot_kin.plot(e_e0_arr, lnjpa_all, pen=None, symbol='o',
                                  symbolBrush='r', symbolSize=8, name='Jp1')
            self.cv_plot_kin.plot(e_e0_arr, lnjpc_all, pen=None, symbol='o',
                                  symbolBrush='b', symbolSize=8, name='Jp2')

    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Help")
        dlg.setMinimumWidth(500)
        layout = QVBoxLayout(dlg)
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
<h3>PySimpleEChem – Quick Start</h3>
<b>Loading files</b><br>
Click <i>Add/Open CV file</i> and choose the appropriate format (VersaSTAT .par,
CorrWare .cor, CSV, or Text). Multiple files can be opened at once; each becomes
a separate entry in the combo box. The most recently added file is activated
automatically.<br><br>

<b>Trim</b><br>
Drag the Trim range slider (or type indices in the position boxes) to restrict
the portion of the CV that is analysed and displayed.<br><br>

<b>Baseline 1 / Baseline 2</b><br>
Set a range on the CV that represents the background current for each peak.
The highlighted segment is fit with a straight line which is then extrapolated
to the peak potential to obtain the net peak current (Jp).<br><br>

<b>Peak search range &amp; Peak positions</b><br>
Set the centre index and half-width of the search window for each peak.
Peak detection modes: <i>max</i> (largest value), <i>min</i> (smallest value),
<i>exact</i> (the index itself), <i>2nd derivative</i> (inflection-point
detection via LOWESS smoothing).<br><br>

<b>Nicholson method</b><br>
Enable when no clear baseline exists for the cathodic peak.  Move the
switching-potential slider to the point on the return sweep where the CV
switches direction.  The corrected cathodic Jp is computed from Nicholson
(1966).<br><br>

<b>Diffusion &amp; Kinetics</b><br>
Load CVs recorded at <i>different scan rates</i>, set the correct scan rate for
each, then enter Bulk concentration and Number of electrons.  The Diffusion tab
plots Jp vs √ν and the Kinetics tab plots ln(Jp) vs (ΔEp/2).  At least two
files are required for a meaningful linear fit.<br><br>

<b>IR compensation</b><br>
Enter the uncompensated resistance (Ω) to correct the voltage axis
(V_corrected = V − I·R).<br><br>

<b>Electrode area</b><br>
Enter the geometric area (cm²) to convert raw current to current density.<br><br>

<b>Report issues / feature requests</b><br>
<a href="https://github.com/kevinsmia1939/PySimpleEChem/issues">
https://github.com/kevinsmia1939/PySimpleEChem/issues</a>
""")
        layout.addWidget(text)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()

    def show_about(self):
        QtWidgets.QMessageBox.about(self, "About PySimpleEChem",
            "<h3>PySimpleEChem</h3>"
            "<p>A simple, open-source GUI for cyclic voltammetry (CV) analysis.</p>"
            "<p><b>Author:</b> Kavin Teenakul</p>"
            "<p><b>Source code:</b> "
            "<a href='https://github.com/kevinsmia1939/PySimpleEChem'>"
            "github.com/kevinsmia1939/PySimpleEChem</a></p>"
            "<p><b>License:</b> GPL-3.0</p>"
            "<p>Features: baseline subtraction, peak detection (max/min/exact/"
            "2nd-derivative), IR compensation, Nicholson method for irreversible "
            "systems, diffusion coefficient (Randles–Ševčík), and reaction rate "
            "constant estimation.</p>")

    def show_abbreviations(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Abbreviations")
        dlg.setMinimumWidth(600)
        layout = QVBoxLayout(dlg)
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        abbrevs = [
            ("Jp1 / Jpa", "Anodic peak current / current density"),
            ("Jp2 / Jpc", "Cathodic peak current / current density"),
            ("Jpc0",      "Cathodic peak current measured against the anodic baseline (Nicholson method)"),
            ("Ep1 / Epa", "Anodic peak potential"),
            ("Ep2 / Epc", "Cathodic peak potential"),
            ("E\u00bd",   "Half-wave potential  (Epa + Epc) / 2"),
            ("\u0394E\u209a", "Peak-to-peak separation  |Epa \u2212 Epc|"),
            ("alpha (\u03b1)", "Charge-transfer coefficient"),
            ("Ano",       "Anodic"),
            ("Cat",       "Cathodic"),
            ("rev",       "Reversible"),
            ("irr",       "Irreversible"),
            ("D (cm\u00b2/s)", "Diffusion coefficient (Randles\u2013\u0160ev\u010d\u00edk equation)"),
            ("k (cm/s)",  "Standard rate constant of the electrode reaction"),
            ("jsp0",      "Current at the switching potential relative to the anodic baseline (Nicholson method)"),
            ("Nicholson method",
             "Nicholson, R. S. Semiempirical Procedure for Measuring with Stationary Electrode "
             "Polarography Rates of Chemical Reactions Involving the Product of Electron Transfer. "
             "Anal. Chem. 1966, 38\u00a0(10), 1406."),
            ("2nd derivative peak detection",
             "Peak detection by second derivatives using LOWESS smoothing. "
             "Data Handling in Science and Technology, Vol.\u00a021, 1998, pp.\u00a0183\u2013190, "
             "DOI: 10.1016/S0922-3487(98)80027-0"),
        ]
        html = "<table style='border-collapse:collapse;' cellpadding='4'>"
        for term, meaning in abbrevs:
            html += (f"<tr><td><b>{term}</b></td>"
                     f"<td style='padding-left:12px;'>{meaning}</td></tr>")
        html += "</table>"
        text.setHtml(html)
        layout.addWidget(text)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()


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
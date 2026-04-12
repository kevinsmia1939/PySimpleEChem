#!/usr/bin/python3
import sys
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, QLabel, QPushButton, QLineEdit, QComboBox, QFileDialog,
    QMenu, QAction, QTabWidget, QTableView, QScrollArea, QDialog,
    QMessageBox, QSplitter, QFrame
)
from PyQt5.QtCore import Qt
from superqt import QRangeSlider
from function_collection import read_cv_format, RDE_kou_lev, linear_fit, check_val

pg.setConfigOptions(antialias=True)

# Colours cycled over loaded files
_LINE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, float):
                col_name = self._data.columns[index.column()]
                if col_name == 'rotation rate':
                    return f"{value:g}"
                return f"{value:.5e}"
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
        self.setWindowTitle("PySimpleRDE — Rotating Disk Electrode Analysis")
        self.create_widgets()
        self.rde_setup_plot()

    # ------------------------------------------------------------------
    # Widget / layout creation
    # ------------------------------------------------------------------
    def create_widgets(self):
        # ── Per-file data storage ──────────────────────────────────────
        self.rde_concat_list_df = pd.DataFrame()      # volt/current columns
        self.rde_df_ir_currden = pd.DataFrame()       # IR-comp + area-norm copy
        self.rde_param_concat_df = pd.DataFrame()     # one row per file
        self.rde_result_display = pd.DataFrame(
            columns=['file name', 'rotation rate', 'unit', 'lim curr (A)'])

        # ── State flags ───────────────────────────────────────────────
        self.empty_rde = True

        # ── Cached arrays for the currently active file ───────────────
        self.rde_chosen_volt = None
        self.rde_chosen_current = None
        self.rde_chosen_idx = 0
        self.rde_chosen_path = None
        self.rde_chosen_data_point_num = 0
        self.trim_start = 0
        self.trim_end = 0
        self.baseline_start = 0
        self.baseline_end = 0
        self.peak_start = 0
        self.peak_end = 0

        # ── Plot overlay items (pre-allocated in rde_setup_plot) ───────
        self.rde_line_artist_list = []
        self.rde_lines = {}            # path → PlotDataItem
        self.rde_plot_baseline = None
        self.rde_plot_peak = None
        self.rde_plot_baseline_fit = None
        self.rde_plot_lim_curr = None

        # ══════════════════════════════════════════════════════════════
        # Central widget & top-level split
        # ══════════════════════════════════════════════════════════════
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=4)

        # ── Left: plot tabs ───────────────────────────────────────────
        self.plot_tab_widget = QTabWidget()
        splitter.addWidget(self.plot_tab_widget)

        tab_rde = QWidget()
        tab_rde_layout = QVBoxLayout(tab_rde)
        self.rde_plot = pg.PlotWidget()
        tab_rde_layout.addWidget(self.rde_plot)
        self.plot_tab_widget.addTab(tab_rde, "RDE")

        tab_kl = QWidget()
        tab_kl_layout = QVBoxLayout(tab_kl)
        self.kl_plot = pg.PlotWidget()
        tab_kl_layout.addWidget(self.kl_plot)
        self.plot_tab_widget.addTab(tab_kl, "Koutecký–Levich")

        # ── Right: controls (scrollable) ──────────────────────────────
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(440)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(ctrl_widget)
        splitter.addWidget(scroll_area)

        # ── File buttons ──────────────────────────────────────────────
        file_row = QHBoxLayout()
        self.rde_addopen_button = QPushButton("Add/Open RDE file", self)
        self.rde_addopen_button.setMenu(self.create_rde_menu())
        self.rde_choose_combo = QComboBox(self)
        self.rde_choose_combo.setFixedWidth(160)
        self.rde_choose_combo.setEditable(False)
        self.rde_choose_combo.setInsertPolicy(QComboBox.NoInsert)
        self.rde_delete_button = QPushButton("Delete RDE", self)
        file_row.addWidget(self.rde_addopen_button)
        file_row.addWidget(self.rde_choose_combo)
        file_row.addWidget(self.rde_delete_button)
        ctrl_layout.addLayout(file_row)

        # ── IR compensation + rotation rate ──────────────────────────
        params_grid = QGridLayout()
        params_grid.addWidget(QLabel("IR compensation (Ω):"), 0, 0)
        self.rde_ircompen_box = QLineEdit("0")
        self.rde_ircompen_box.setEnabled(False)
        params_grid.addWidget(self.rde_ircompen_box, 0, 1)

        params_grid.addWidget(QLabel("Electrode area (cm²):"), 1, 0)
        self.rde_elec_area_box = QLineEdit("1")
        self.rde_elec_area_box.setEnabled(False)
        params_grid.addWidget(self.rde_elec_area_box, 1, 1)

        params_grid.addWidget(QLabel("Rotation rate:"), 2, 0)
        ror_row = QHBoxLayout()
        self.rde_ror_box = QLineEdit("0")
        self.rde_ror_box.setEnabled(False)
        self.rde_ror_unit_combo = QComboBox()
        self.rde_ror_unit_combo.addItems(["RPM", "rad/s"])
        self.rde_ror_unit_combo.setEnabled(False)
        ror_row.addWidget(self.rde_ror_box)
        ror_row.addWidget(self.rde_ror_unit_combo)
        ror_container = QWidget()
        ror_container.setLayout(ror_row)
        params_grid.addWidget(ror_container, 2, 1)
        ctrl_layout.addLayout(params_grid)

        ctrl_layout.addWidget(self._hline())

        # ── Trim slider ───────────────────────────────────────────────
        ctrl_layout.addWidget(QLabel("Trim:"))
        self.rde_trim_slider = QRangeSlider(Qt.Horizontal)
        self.rde_trim_slider.setMinimum(0)
        self.rde_trim_slider.setMaximum(1)
        self.rde_trim_slider.setValue((0, 1))
        self.rde_trim_slider.setEnabled(False)
        trim_box_row = QHBoxLayout()
        trim_box_row.addWidget(QLabel("Start:"))
        self.rde_trim_start_box = QLineEdit("0")
        self.rde_trim_start_box.setFixedWidth(60)
        self.rde_trim_start_box.setEnabled(False)
        trim_box_row.addWidget(self.rde_trim_start_box)
        trim_box_row.addWidget(QLabel("End:"))
        self.rde_trim_end_box = QLineEdit("0")
        self.rde_trim_end_box.setFixedWidth(60)
        self.rde_trim_end_box.setEnabled(False)
        trim_box_row.addWidget(self.rde_trim_end_box)
        ctrl_layout.addWidget(self.rde_trim_slider)
        ctrl_layout.addLayout(trim_box_row)

        ctrl_layout.addWidget(self._hline())

        # ── Baseline range slider ─────────────────────────────────────
        ctrl_layout.addWidget(QLabel("Baseline region:"))
        self.rde_baseline_slider = QRangeSlider(Qt.Horizontal)
        self.rde_baseline_slider.setMinimum(0)
        self.rde_baseline_slider.setMaximum(1)
        self.rde_baseline_slider.setValue((0, 0))
        self.rde_baseline_slider.setEnabled(False)
        bl_box_row = QHBoxLayout()
        bl_box_row.addWidget(QLabel("Start:"))
        self.rde_bl_start_box = QLineEdit("0")
        self.rde_bl_start_box.setFixedWidth(60)
        self.rde_bl_start_box.setEnabled(False)
        bl_box_row.addWidget(self.rde_bl_start_box)
        bl_box_row.addWidget(QLabel("End:"))
        self.rde_bl_end_box = QLineEdit("0")
        self.rde_bl_end_box.setFixedWidth(60)
        self.rde_bl_end_box.setEnabled(False)
        bl_box_row.addWidget(self.rde_bl_end_box)
        ctrl_layout.addWidget(self.rde_baseline_slider)
        ctrl_layout.addLayout(bl_box_row)

        ctrl_layout.addWidget(self._hline())

        # ── Peak / plateau slider ─────────────────────────────────────
        ctrl_layout.addWidget(QLabel("Plateau baseline region:"))
        self.rde_peak_slider = QRangeSlider(Qt.Horizontal)
        self.rde_peak_slider.setMinimum(0)
        self.rde_peak_slider.setMaximum(1)
        self.rde_peak_slider.setValue((0, 0))
        self.rde_peak_slider.setEnabled(False)
        pk_box_row = QHBoxLayout()
        pk_box_row.addWidget(QLabel("Start:"))
        self.rde_pk_start_box = QLineEdit("0")
        self.rde_pk_start_box.setFixedWidth(60)
        self.rde_pk_start_box.setEnabled(False)
        pk_box_row.addWidget(self.rde_pk_start_box)
        pk_box_row.addWidget(QLabel("End:"))
        self.rde_pk_end_box = QLineEdit("0")
        self.rde_pk_end_box.setFixedWidth(60)
        self.rde_pk_end_box.setEnabled(False)
        pk_box_row.addWidget(self.rde_pk_end_box)
        ctrl_layout.addWidget(self.rde_peak_slider)
        ctrl_layout.addLayout(pk_box_row)

        ctrl_layout.addWidget(self._hline())

        # ── Diffusion / kinetics parameters ──────────────────────────
        phys_grid = QGridLayout()
        phys_grid.addWidget(QLabel("Number of electrons (n):"), 0, 0)
        self.rde_elec_n_box = QLineEdit("1")
        self.rde_elec_n_box.setEnabled(False)
        phys_grid.addWidget(self.rde_elec_n_box, 0, 1)

        phys_grid.addWidget(QLabel("Kinematic viscosity (cm²/s):"), 1, 0)
        self.rde_kinvis_box = QLineEdit("0.01")
        self.rde_kinvis_box.setEnabled(False)
        phys_grid.addWidget(self.rde_kinvis_box, 1, 1)

        phys_grid.addWidget(QLabel("Bulk concentration (mol/cm³):"), 2, 0)
        self.rde_bulk_conc_box = QLineEdit("0")
        self.rde_bulk_conc_box.setEnabled(False)
        phys_grid.addWidget(self.rde_bulk_conc_box, 2, 1)
        ctrl_layout.addLayout(phys_grid)

        ctrl_layout.addWidget(self._hline())

        # ── Calculated outputs ────────────────────────────────────────
        out_grid = QGridLayout()
        out_grid.addWidget(QLabel("Diffusion coeff. D (cm²/s):"), 0, 0)
        self.rde_D_display = QLineEdit("")
        self.rde_D_display.setReadOnly(True)
        out_grid.addWidget(self.rde_D_display, 0, 1)

        out_grid.addWidget(QLabel("Kinetic current I\u2096 (A):"), 1, 0)
        self.rde_jkin_display = QLineEdit("")
        self.rde_jkin_display.setReadOnly(True)
        out_grid.addWidget(self.rde_jkin_display, 1, 1)

        out_grid.addWidget(QLabel("K–L fit R²:"), 2, 0)
        self.rde_r2_display = QLineEdit("")
        self.rde_r2_display.setReadOnly(True)
        out_grid.addWidget(self.rde_r2_display, 2, 1)
        ctrl_layout.addLayout(out_grid)

        ctrl_layout.addWidget(self._hline())

        # ── Utility buttons ───────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.help_button = QPushButton("Help", self)
        self.about_button = QPushButton("About", self)
        self.abbrev_button = QPushButton("What is this?", self)
        btn_row.addWidget(self.help_button)
        btn_row.addWidget(self.about_button)
        btn_row.addWidget(self.abbrev_button)
        ctrl_layout.addLayout(btn_row)

        # ── Bottom: results table ─────────────────────────────────────
        self.rde_result_table = QtWidgets.QTableView()
        self.rde_result_table.setModel(TableModel(self.rde_result_display))
        self.rde_result_table.setFixedHeight(150)
        self.rde_copy_button = QPushButton("Copy results")
        self.rde_copy_button.setEnabled(False)
        result_label_row = QHBoxLayout()
        result_label_row.addWidget(QLabel("Results:"))
        result_label_row.addStretch()
        result_label_row.addWidget(self.rde_copy_button)
        main_layout.addLayout(result_label_row)
        main_layout.addWidget(self.rde_result_table)

        # ── Signal connections ────────────────────────────────────────
        # Trim slider: draw all on move, full redraw on release
        self.rde_trim_slider.sliderMoved.connect(self.rde_draw_all_rde)
        self.rde_trim_start_box.textChanged.connect(self.rde_draw_all_rde)
        self.rde_trim_end_box.textChanged.connect(self.rde_draw_all_rde)

        # Baseline/peak sliders: fast highlight on move, compute on release
        self.rde_baseline_slider.sliderMoved.connect(self.rde_draw_overlay)
        self.rde_baseline_slider.sliderReleased.connect(self.rde_compute_and_update)
        self.rde_bl_start_box.textChanged.connect(self.rde_draw_overlay_from_box)
        self.rde_bl_end_box.textChanged.connect(self.rde_draw_overlay_from_box)

        self.rde_peak_slider.sliderMoved.connect(self.rde_draw_overlay)
        self.rde_peak_slider.sliderReleased.connect(self.rde_compute_and_update)
        self.rde_pk_start_box.textChanged.connect(self.rde_draw_overlay_from_box)
        self.rde_pk_end_box.textChanged.connect(self.rde_draw_overlay_from_box)

        # Physical/instrument parameters trigger full recompute on Enter
        self.rde_ircompen_box.returnPressed.connect(self.rde_modify_params)
        self.rde_elec_area_box.returnPressed.connect(self.rde_modify_params)
        self.rde_ror_box.returnPressed.connect(self.rde_save_ror)
        self.rde_ror_unit_combo.currentIndexChanged.connect(self.rde_save_ror)
        self.rde_elec_n_box.returnPressed.connect(self.rde_calc_kou_lev)
        self.rde_kinvis_box.returnPressed.connect(self.rde_calc_kou_lev)
        self.rde_bulk_conc_box.returnPressed.connect(self.rde_calc_kou_lev)

        self.rde_choose_combo.textActivated.connect(self.rde_open_switch_rde)
        self.rde_delete_button.clicked.connect(self.rde_delete_rde)
        self.rde_copy_button.clicked.connect(self.rde_copy_results)
        self.help_button.clicked.connect(self.show_help)
        self.about_button.clicked.connect(self.show_about)
        self.abbrev_button.clicked.connect(self.show_abbreviations)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def _clamp_int(self, text, lo, hi, fallback):
        try:
            v = int(text)
            return max(lo, min(hi, v))
        except (ValueError, TypeError):
            return fallback

    # ------------------------------------------------------------------
    # Plot setup
    # ------------------------------------------------------------------
    def rde_setup_plot(self):
        self.rde_plot.setLabel('left', text='Current (A)')
        self.rde_plot.setLabel('bottom', text='Voltage (V)')
        self.rde_plot.addLegend()

        self.kl_plot.setLabel('left', text='1/I (A⁻¹)')
        self.kl_plot.setLabel('bottom', text='ω⁻¹/² (rad/s)⁻¹/²')
        self.kl_plot.addLegend()

        # Pre-allocate overlay items (setData used instead of remove/add)
        self.rde_plot_baseline = self.rde_plot.plot(
            [], [], pen=pg.mkPen(color='red', width=4))
        self.rde_plot_peak = self.rde_plot.plot(
            [], [], pen=pg.mkPen(color='cyan', width=4))
        self.rde_plot_baseline_fit = self.rde_plot.plot(
            [], [], pen=pg.mkPen(color='white', width=1,
                                 style=QtCore.Qt.DashLine))
        self.rde_plot_peak_fit = self.rde_plot.plot(
            [], [], pen=pg.mkPen(color='cyan', width=1,
                                 style=QtCore.Qt.DashLine))
        self.rde_plot_lim_curr = self.rde_plot.plot(
            [], [], pen=pg.mkPen(color='yellow', width=2,
                                 style=QtCore.Qt.DashLine))
        self.rde_plot_cross_marker = self.rde_plot.plot(
            [], [], pen=None, symbol='o', symbolSize=10,
            symbolBrush=pg.mkBrush('yellow'), symbolPen=pg.mkPen(None))

        # K-L overlay
        self.kl_plot_fit = self.kl_plot.plot(
            [], [], pen=pg.mkPen(color='white', width=2),
            name='K–L fit')

    # ------------------------------------------------------------------
    # File menu
    # ------------------------------------------------------------------
    def create_rde_menu(self):
        menu = QMenu(self)
        for label, ext in [('VersaSTAT (.par)', '.par'),
                            ('CorrWare (.cor)',  '.cor'),
                            ('CSV (.csv)',        '.csv'),
                            ('Text (.txt)',       '.txt')]:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, e=ext: self.rde_open_file(e))
            menu.addAction(action)
        return menu

    # ------------------------------------------------------------------
    # Open / load files
    # ------------------------------------------------------------------
    def rde_open_file(self, ext):
        fmt_map = {'.par': 'VersaSTAT', '.cor': 'CorrWare',
                   '.csv': 'CSV',        '.txt': 'text'}
        fmt = fmt_map.get(ext, 'CSV')

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open RDE file", "",
            f"RDE Files (*{ext})")
        if not paths:
            return

        for path in paths:
            cv_df, _, _ = read_cv_format(path, fmt)
            n_pts = cv_df.shape[0]

            # Store volt/current data
            self.rde_concat_list_df = pd.concat(
                [self.rde_concat_list_df, cv_df], axis=1)
            self.rde_df_ir_currden = pd.concat(
                [self.rde_df_ir_currden, cv_df], axis=1)

            # Default parameters for this file
            param_row = pd.DataFrame([{
                'file path':      path,
                'file name':      path.split('/')[-1],
                'file format':    fmt,
                'n_data_points':  n_pts,
                'trim_start':     0,
                'trim_end':       n_pts - 1,
                'baseline_start': 0,
                'baseline_end':   int(n_pts * 0.1),
                'peak_start':     int(n_pts * 0.7),
                'peak_end':       n_pts - 1,
                'ir_compensation': 0.0,
                'elec_area':      1.0,
                'rotation_rate':  0.0,
                'ror_unit':       'RPM',
            }])
            self.rde_param_concat_df = pd.concat(
                [self.rde_param_concat_df, param_row],
                axis=0, ignore_index=True)

            # Placeholder result row
            result_row = pd.DataFrame([{
                'file name':        path.split('/')[-1],
                'rotation rate':    '-',
                'unit':             'RPM',
                'lim curr (A)':     '-',
            }])
            self.rde_result_display = pd.concat(
                [self.rde_result_display, result_row],
                axis=0, ignore_index=True)

        # Update combo box
        self.rde_choose_combo.clear()
        self.rde_choose_combo.addItems(
            self.rde_param_concat_df['file name'].astype(str).tolist())

        # Enable all controls
        if not self.rde_param_concat_df.empty:
            self.empty_rde = False
            self._set_controls_enabled(True)

        # Switch to the newly added file
        last_idx = len(self.rde_param_concat_df) - 1
        self.rde_choose_combo.setCurrentIndex(last_idx)
        self.rde_open_switch_rde()
        self.rde_draw_all_rde()

    # ------------------------------------------------------------------
    # Delete current file
    # ------------------------------------------------------------------
    def rde_delete_rde(self):
        if self.empty_rde or self.rde_choose_combo.count() == 0:
            return

        del_idx = int(self.rde_choose_combo.currentIndex())
        del_path = self.rde_param_concat_df.loc[del_idx, 'file path']

        for col in [str(del_path) + ' volt', str(del_path) + ' current']:
            for df in [self.rde_concat_list_df, self.rde_df_ir_currden]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

        self.rde_param_concat_df.drop(index=del_idx, inplace=True)
        self.rde_param_concat_df.reset_index(drop=True, inplace=True)
        self.rde_result_display.drop(index=del_idx, inplace=True)
        self.rde_result_display.reset_index(drop=True, inplace=True)

        # Remove the plot line for this file
        if del_path in self.rde_lines:
            self.rde_plot.removeItem(self.rde_lines.pop(del_path))

        self.rde_choose_combo.clear()
        if self.rde_param_concat_df.empty:
            self.empty_rde = True
            self._set_controls_enabled(False)
            self.rde_plot_baseline.setData([], [])
            self.rde_plot_peak.setData([], [])
            self.rde_plot_baseline_fit.setData([], [])
            self.rde_plot_peak_fit.setData([], [])
            self.rde_plot_lim_curr.setData([], [])
            self.rde_plot_cross_marker.setData([], [])
            self.kl_plot.clear()
            self.kl_plot_fit = self.kl_plot.plot([], [], pen=pg.mkPen('white', width=2))
            self.rde_D_display.setText("")
            self.rde_jkin_display.setText("")
            self.rde_r2_display.setText("")
        else:
            self.rde_choose_combo.addItems(
                self.rde_param_concat_df['file name'].astype(str).tolist())
            new_idx = min(del_idx, len(self.rde_param_concat_df) - 1)
            self.rde_choose_combo.setCurrentIndex(new_idx)
            self.rde_open_switch_rde()
            self.rde_draw_all_rde()

        self.rde_result_table.setModel(TableModel(self.rde_result_display))

    # ------------------------------------------------------------------
    # Switch active file
    # ------------------------------------------------------------------
    def rde_open_switch_rde(self):
        if self.empty_rde or self.rde_choose_combo.count() == 0:
            return

        self.rde_chosen_idx = int(self.rde_choose_combo.currentIndex())
        row = self.rde_param_concat_df.loc[self.rde_chosen_idx]
        self.rde_chosen_path = row['file path']
        self.rde_chosen_data_point_num = int(row['n_data_points'])

        # Refresh cached numpy arrays (reflects IR / area changes)
        self.rde_chosen_volt = np.array(
            self.rde_df_ir_currden[str(self.rde_chosen_path) + ' volt'])
        self.rde_chosen_current = np.array(
            self.rde_df_ir_currden[str(self.rde_chosen_path) + ' current'])

        self.rde_set_slider_val()

        # Restore scalar controls without triggering saves
        self.rde_ircompen_box.setText(str(row['ir_compensation']))
        self.rde_elec_area_box.setText(str(row['elec_area']))
        self.rde_ror_box.setText(str(row['rotation_rate']))
        idx_unit = self.rde_ror_unit_combo.findText(str(row['ror_unit']))
        if idx_unit >= 0:
            self.rde_ror_unit_combo.blockSignals(True)
            self.rde_ror_unit_combo.setCurrentIndex(idx_unit)
            self.rde_ror_unit_combo.blockSignals(False)

        self.rde_draw_overlay()
        self.rde_compute_and_update()

    def rde_set_slider_val(self):
        row = self.rde_param_concat_df.loc[self.rde_chosen_idx]
        max_idx = self.rde_chosen_data_point_num - 1

        self.trim_start   = int(row['trim_start'])
        self.trim_end     = int(row['trim_end'])
        self.baseline_start = int(row['baseline_start'])
        self.baseline_end   = int(row['baseline_end'])
        self.peak_start   = int(row['peak_start'])
        self.peak_end     = int(row['peak_end'])

        for slider in (self.rde_trim_slider,
                       self.rde_baseline_slider,
                       self.rde_peak_slider):
            slider.setMaximum(max_idx)

        for slider, lo, hi in [
            (self.rde_trim_slider,     self.trim_start,     self.trim_end),
            (self.rde_baseline_slider, self.baseline_start, self.baseline_end),
            (self.rde_peak_slider,     self.peak_start,     self.peak_end),
        ]:
            slider.blockSignals(True)
            slider.setValue((lo, hi))
            slider.blockSignals(False)

        for box, val in [
            (self.rde_trim_start_box,  self.trim_start),
            (self.rde_trim_end_box,    self.trim_end),
            (self.rde_bl_start_box,    self.baseline_start),
            (self.rde_bl_end_box,      self.baseline_end),
            (self.rde_pk_start_box,    self.peak_start),
            (self.rde_pk_end_box,      self.peak_end),
        ]:
            box.blockSignals(True)
            box.setText(str(val))
            box.blockSignals(False)

    # ------------------------------------------------------------------
    # IR compensation / electrode area changed
    # ------------------------------------------------------------------
    def rde_modify_params(self):
        if self.empty_rde:
            return
        try:
            ir = float(self.rde_ircompen_box.text())
        except ValueError:
            ir = 0.0
        try:
            area = float(self.rde_elec_area_box.text())
            if area <= 0:
                area = 1.0
        except ValueError:
            area = 1.0

        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'ir_compensation'] = ir
        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'elec_area'] = area

        path = self.rde_chosen_path
        raw_volt = np.array(self.rde_concat_list_df[path + ' volt'])
        raw_curr = np.array(self.rde_concat_list_df[path + ' current'])
        self.rde_df_ir_currden[path + ' volt']    = raw_volt + raw_curr * ir
        self.rde_df_ir_currden[path + ' current'] = raw_curr / area

        # Refresh cache
        self.rde_chosen_volt    = np.array(self.rde_df_ir_currden[path + ' volt'])
        self.rde_chosen_current = np.array(self.rde_df_ir_currden[path + ' current'])

        self.rde_draw_all_rde()
        self.rde_compute_and_update()

    def rde_save_ror(self):
        if self.empty_rde:
            return
        try:
            ror = float(self.rde_ror_box.text())
        except ValueError:
            ror = 0.0
        unit = self.rde_ror_unit_combo.currentText()
        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'rotation_rate'] = ror
        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'ror_unit'] = unit
        self.rde_result_display.loc[self.rde_chosen_idx, 'rotation rate'] = ror
        self.rde_result_display.loc[self.rde_chosen_idx, 'unit'] = unit
        self.rde_result_table.setModel(TableModel(self.rde_result_display))
        self.rde_calc_kou_lev()

    # ------------------------------------------------------------------
    # Draw ALL loaded LSV lines
    # ------------------------------------------------------------------
    def rde_draw_all_rde(self):
        if self.empty_rde:
            return

        max_idx = self.rde_chosen_data_point_num - 1
        sender = self.sender()

        if sender is self.rde_trim_slider:
            self.trim_start = self.rde_trim_slider.value()[0]
            self.trim_end   = self.rde_trim_slider.value()[1]
            for box, val in [(self.rde_trim_start_box, self.trim_start),
                             (self.rde_trim_end_box,   self.trim_end)]:
                box.blockSignals(True)
                box.setText(str(val))
                box.blockSignals(False)
        else:
            self.trim_start = self._clamp_int(
                self.rde_trim_start_box.text(), 0, max_idx, self.trim_start)
            self.trim_end   = self._clamp_int(
                self.rde_trim_end_box.text(),   0, max_idx, self.trim_end)
            self.rde_trim_slider.blockSignals(True)
            self.rde_trim_slider.setValue((self.trim_start, self.trim_end))
            self.rde_trim_slider.blockSignals(False)

        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'trim_start'] = self.trim_start
        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'trim_end']   = self.trim_end

        # Redraw every loaded file's trimmed line
        for i, row in self.rde_param_concat_df.iterrows():
            path  = row['file path']
            ts    = int(row['trim_start'])
            te    = int(row['trim_end'])
            volt  = np.array(self.rde_df_ir_currden[path + ' volt'])[ts:te]
            curr  = np.array(self.rde_df_ir_currden[path + ' current'])[ts:te]
            color = _LINE_COLORS[i % len(_LINE_COLORS)]
            name  = row['file name']
            if path in self.rde_lines:
                self.rde_lines[path].setData(volt, curr)
            else:
                item = self.rde_plot.plot(
                    volt, curr,
                    pen=pg.mkPen(color=color, width=2),
                    name=name)
                self.rde_lines[path] = item

        self.rde_draw_overlay()

    # ------------------------------------------------------------------
    # Draw baseline + peak highlights (fast — called on sliderMoved)
    # ------------------------------------------------------------------
    def rde_draw_overlay(self):
        if self.empty_rde:
            return

        sender = self.sender()
        max_idx = self.rde_chosen_data_point_num - 1

        if sender is self.rde_baseline_slider:
            self.baseline_start = self.rde_baseline_slider.value()[0]
            self.baseline_end   = self.rde_baseline_slider.value()[1]
            for box, val in [(self.rde_bl_start_box, self.baseline_start),
                             (self.rde_bl_end_box,   self.baseline_end)]:
                box.blockSignals(True)
                box.setText(str(val))
                box.blockSignals(False)
        elif sender is self.rde_peak_slider:
            self.peak_start = self.rde_peak_slider.value()[0]
            self.peak_end   = self.rde_peak_slider.value()[1]
            for box, val in [(self.rde_pk_start_box, self.peak_start),
                             (self.rde_pk_end_box,   self.peak_end)]:
                box.blockSignals(True)
                box.setText(str(val))
                box.blockSignals(False)

        volt    = self.rde_chosen_volt
        current = self.rde_chosen_current

        self.rde_plot_baseline.setData(
            volt[self.baseline_start:self.baseline_end],
            current[self.baseline_start:self.baseline_end])
        self.rde_plot_peak.setData(
            volt[self.peak_start:self.peak_end],
            current[self.peak_start:self.peak_end])

    def rde_draw_overlay_from_box(self):
        """Called when text boxes change — sync sliders then draw."""
        if self.empty_rde:
            return
        max_idx = self.rde_chosen_data_point_num - 1

        self.baseline_start = self._clamp_int(
            self.rde_bl_start_box.text(), 0, max_idx, self.baseline_start)
        self.baseline_end   = self._clamp_int(
            self.rde_bl_end_box.text(),   0, max_idx, self.baseline_end)
        self.peak_start = self._clamp_int(
            self.rde_pk_start_box.text(), 0, max_idx, self.peak_start)
        self.peak_end   = self._clamp_int(
            self.rde_pk_end_box.text(),   0, max_idx, self.peak_end)

        self.rde_baseline_slider.blockSignals(True)
        self.rde_baseline_slider.setValue((self.baseline_start, self.baseline_end))
        self.rde_baseline_slider.blockSignals(False)
        self.rde_peak_slider.blockSignals(True)
        self.rde_peak_slider.setValue((self.peak_start, self.peak_end))
        self.rde_peak_slider.blockSignals(False)

        self.rde_draw_overlay()
        self.rde_compute_and_update()

    # ------------------------------------------------------------------
    # Compute limiting current for the active file, update K-L plot
    # ------------------------------------------------------------------
    def rde_compute_and_update(self):
        if self.empty_rde:
            return

        volt    = self.rde_chosen_volt
        current = self.rde_chosen_current

        # Save slider positions to param DataFrame
        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'baseline_start'] = self.baseline_start
        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'baseline_end']   = self.baseline_end
        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'peak_start']     = self.peak_start
        self.rde_param_concat_df.loc[self.rde_chosen_idx, 'peak_end']       = self.peak_end

        # Baseline linear fit
        bl_volt = volt[self.baseline_start:self.baseline_end]
        bl_curr = current[self.baseline_start:self.baseline_end]
        if len(bl_volt) < 2 or len(bl_volt) != len(bl_curr):
            return
        try:
            _, bl_poly = linear_fit(bl_volt, bl_curr)
        except Exception:
            return

        # Plateau baseline linear fit
        pk_volt = volt[self.peak_start:self.peak_end]
        pk_curr = current[self.peak_start:self.peak_end]
        if len(pk_volt) < 2 or len(pk_volt) != len(pk_curr):
            return
        try:
            _, pk_poly = linear_fit(pk_volt, pk_curr)
        except Exception:
            return

        # Extrapolate both lines across the full wave region
        x_fit_start = volt[self.baseline_start]
        x_fit_end   = volt[min(self.peak_end, len(volt) - 1)]
        self.rde_plot_baseline_fit.setData(
            [x_fit_start, x_fit_end],
            [bl_poly(x_fit_start), bl_poly(x_fit_end)])
        self.rde_plot_peak_fit.setData(
            [x_fit_start, x_fit_end],
            [pk_poly(x_fit_start), pk_poly(x_fit_end)])

        # Find where the actual data crosses the midpoint of the two
        # extrapolated lines: mid(V) = (bl_poly(V) + pk_poly(V)) / 2
        # True limiting current = pk_poly(V_cross) - bl_poly(V_cross)
        ts = self.trim_start
        te = self.trim_end
        v_trim = volt[ts:te]
        i_trim = current[ts:te]
        if len(v_trim) < 2:
            return
        diff = i_trim - (bl_poly(v_trim) + pk_poly(v_trim)) / 2.0

        v_cross = None
        for k in range(len(diff) - 1):
            if diff[k] * diff[k + 1] <= 0:
                denom = diff[k] - diff[k + 1]
                if denom != 0:
                    t = diff[k] / denom
                    v_cross = v_trim[k] + t * (v_trim[k + 1] - v_trim[k])
                else:
                    v_cross = (v_trim[k] + v_trim[k + 1]) / 2.0
                break

        if v_cross is None:
            self.rde_plot_lim_curr.setData([], [])
            self.rde_plot_cross_marker.setData([], [])
            return

        lim_curr = pk_poly(v_cross) - bl_poly(v_cross)
        i_cross = (bl_poly(v_cross) + pk_poly(v_cross)) / 2.0

        # Vertical marker from baseline line to plateau line at V_cross
        self.rde_plot_lim_curr.setData(
            [v_cross, v_cross],
            [bl_poly(v_cross), pk_poly(v_cross)])

        # Dot on the data at the crossing point
        self.rde_plot_cross_marker.setData([v_cross], [i_cross])

        # Save to results
        self.rde_result_display.loc[self.rde_chosen_idx, 'lim curr (A)'] = lim_curr
        self.rde_result_table.setModel(TableModel(self.rde_result_display))

        self.rde_calc_kou_lev()

    # ------------------------------------------------------------------
    # Koutecký–Levich analysis across all loaded files
    # ------------------------------------------------------------------
    def rde_calc_kou_lev(self):
        if self.empty_rde:
            return

        # Gather numeric rows
        try:
            lim_curr_series = pd.to_numeric(
                self.rde_result_display['lim curr (A)'], errors='coerce')
            ror_series = pd.to_numeric(
                self.rde_param_concat_df['rotation_rate'], errors='coerce')
        except Exception:
            return

        valid_mask = lim_curr_series.notna() & ror_series.notna() & (ror_series > 0)
        if valid_mask.sum() < 2:
            return

        lim_curr_arr = lim_curr_series[valid_mask].values
        ror_arr      = ror_series[valid_mask].values
        unit_arr     = self.rde_param_concat_df.loc[valid_mask, 'ror_unit'].values

        try:
            bulk_conc = float(self.rde_bulk_conc_box.text())
        except ValueError:
            bulk_conc = 0.0
        try:
            n = int(self.rde_elec_n_box.text())
        except ValueError:
            n = 1
        try:
            kinvis = float(self.rde_kinvis_box.text())
            if kinvis <= 0:
                kinvis = 0.01
        except ValueError:
            kinvis = 0.01

        try:
            inv_sqrt_ror, j_inv_fit, D, j_kin, kl_poly, r2 = RDE_kou_lev(
                ror_arr, lim_curr_arr, bulk_conc, n, kinvis, unit_arr)
        except Exception:
            return

        # Update K-L scatter points per file
        self.kl_plot.clear()
        self.kl_plot_fit = self.kl_plot.plot(
            [], [], pen=pg.mkPen('white', width=2), name='K–L fit')

        for j, idx in enumerate(self.rde_param_concat_df[valid_mask].index):
            row   = self.rde_param_concat_df.loc[idx]
            unit  = row['ror_unit']
            conv  = 0.104719755 if unit == 'RPM' else 1.0
            omega = float(row['rotation_rate']) * conv
            if omega <= 0:
                continue
            inv_w = 1.0 / np.sqrt(omega)
            inv_i = 1.0 / float(self.rde_result_display.loc[idx, 'lim curr (A)'])
            color = _LINE_COLORS[idx % len(_LINE_COLORS)]
            self.kl_plot.plot(
                [inv_w], [inv_i],
                pen=None, symbol='o', symbolSize=8,
                symbolBrush=pg.mkBrush(color),
                name=row['file name'])

        # Draw fit line
        if len(inv_sqrt_ror) >= 2:
            x_line = np.linspace(min(inv_sqrt_ror), max(inv_sqrt_ror), 100)
            self.kl_plot_fit.setData(x_line, kl_poly(x_line))

        # Display results
        if np.isfinite(D):
            self.rde_D_display.setText(f"{D:.4e}")
        else:
            self.rde_D_display.setText("N/A (need bulk conc > 0)")
        self.rde_jkin_display.setText(f"{j_kin:.4e}")
        self.rde_r2_display.setText(f"{r2:.4f}")

    # ------------------------------------------------------------------
    # Enable / disable all interactive controls
    # ------------------------------------------------------------------
    def _set_controls_enabled(self, state: bool):
        for w in (self.rde_ircompen_box, self.rde_elec_area_box,
                  self.rde_ror_box, self.rde_ror_unit_combo,
                  self.rde_trim_slider, self.rde_trim_start_box,
                  self.rde_trim_end_box,
                  self.rde_baseline_slider, self.rde_bl_start_box,
                  self.rde_bl_end_box,
                  self.rde_peak_slider, self.rde_pk_start_box,
                  self.rde_pk_end_box,
                  self.rde_elec_n_box, self.rde_kinvis_box,
                  self.rde_bulk_conc_box,
                  self.rde_copy_button):
            w.setEnabled(state)

    def rde_copy_results(self):
        if self.rde_result_display.empty:
            return
        QtWidgets.QApplication.clipboard().setText(
            self.rde_result_display.to_csv(sep='\t', index=False, na_rep='')
        )

    # ------------------------------------------------------------------
    # Help / About / Abbreviations dialogs
    # ------------------------------------------------------------------
    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Help — PySimpleRDE")
        dlg.resize(500, 400)
        layout = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        text = QLabel("""
<h3>PySimpleRDE — Quick Start</h3>
<ol>
<li><b>Add/Open RDE file</b> — load one or more LSV files recorded at different rotation rates.</li>
<li>Enter the <b>rotation rate</b> for each file (switch files using the combo box).</li>
<li>Set the <b>Trim</b> slider to cut noisy ends of the scan.</li>
<li>Set the <b>Baseline</b> slider over the pre-wave (no reaction) region — a linear fit (white dashed) is extrapolated from here.</li>
<li>Set the <b>Plateau baseline</b> slider over the diffusion-limited plateau region — a second linear fit (cyan dashed) is extrapolated from here.</li>
<li>The true limiting current is measured at the voltage where the actual data crosses the midpoint of the two extrapolated lines (half-wave point). The yellow vertical marker shows the separation between the two lines at that crossing, which is the true limiting current.</li>
<li>Enter <b>n</b> (electrons transferred), <b>kinematic viscosity</b> ν, and <b>bulk concentration</b> C to obtain the diffusion coefficient D.</li>
<li>The <b>Koutecký–Levich</b> tab shows 1/I vs ω<sup>−½</sup>; the y-intercept gives 1/I<sub>k</sub>.</li>
</ol>
<p>At least 2 files with different rotation rates are required for the K–L plot.</p>
""")
        text.setWordWrap(True)
        scroll.setWidget(text)
        layout.addWidget(scroll)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()

    def show_about(self):
        QMessageBox.about(self, "About PySimpleRDE",
            "PySimpleRDE\n\n"
            "Rotating Disk Electrode analysis GUI\n"
            "Part of the PySimpleEChem project\n\n"
            "Written by Kavin Teenakul\n"
            "License: GPLv3\n"
            "https://github.com/kevinsmia1939/PySimpleEChem")

    def show_abbreviations(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Abbreviations — PySimpleRDE")
        dlg.resize(480, 360)
        layout = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        text = QLabel("""
<table>
<tr><th align='left'>Abbrev.</th><th align='left'>Meaning</th></tr>
<tr><td>RDE</td><td>Rotating Disk Electrode</td></tr>
<tr><td>LSV</td><td>Linear Sweep Voltammetry</td></tr>
<tr><td>I<sub>L</sub></td><td>Limiting current (A)</td></tr>
<tr><td>I<sub>k</sub></td><td>Kinetic current (A)</td></tr>
<tr><td>D</td><td>Diffusion coefficient (cm²/s)</td></tr>
<tr><td>n</td><td>Number of electrons transferred</td></tr>
<tr><td>F</td><td>Faraday constant (96485 C/mol)</td></tr>
<tr><td>ν (kinvis)</td><td>Kinematic viscosity (cm²/s)</td></tr>
<tr><td>C (bulk conc)</td><td>Bulk concentration (mol/cm³)</td></tr>
<tr><td>ω</td><td>Angular velocity (rad/s)</td></tr>
<tr><td>RPM</td><td>Revolutions per minute (1 RPM = 0.10472 rad/s)</td></tr>
<tr><td>K–L</td><td>Koutecký–Levich (1/I vs ω<sup>−½</sup> plot)</td></tr>
<tr><td>R²</td><td>Coefficient of determination of the K–L linear fit</td></tr>
<tr><td>IR comp.</td><td>IR (ohmic) compensation applied to voltage</td></tr>
</table>
<br/>
<b>Levich equation:</b> I<sub>L</sub> = 0.62 n F D<sup>2/3</sup> ω<sup>½</sup> ν<sup>−1/6</sup> C<br/>
<b>Koutecký–Levich:</b> 1/I = 1/I<sub>k</sub> + 1/(B ω<sup>½</sup>)
&nbsp; where B = 0.62 n F D<sup>2/3</sup> ν<sup>−1/6</sup> C
""")
        text.setWordWrap(True)
        scroll.setWidget(text)
        layout.addWidget(scroll)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()


# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

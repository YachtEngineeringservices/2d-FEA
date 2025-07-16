import sys
import logging
import json
import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QTextEdit,
    QProgressDialog, QTabWidget, QMenuBar, QMenu, QListWidget, QListWidgetItem,
    QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox
)
from PySide6.QtGui import QAction
import dolfinx
from dolfinx import fem
import dolfinx.plot
import numpy as np
from .mpl_canvas import MplCanvas
from fea import meshing, solver

# Get a logger for this module
log = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """
    The main window for the 2D FEA application.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Torsional Analysis FEA")
        self.setGeometry(100, 100, 1200, 800) # Increased width for tabs
        log.info("Main window initialized.")

        # Data storage
        self.outer_points = []
        self.inner_points = []
        self.active_points_list = self.outer_points # Start with outer selected
        self.current_file_path = None  # Track current saved file
        self.selected_point_index = -1  # Track selected point for editing

        # Create menu bar
        self.create_menu_bar()

        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # --- Plotting Canvas (initialize early) ---
        self.plot_canvas = MplCanvas(self)
        self.plot_canvas.mpl_connect('button_press_event', self.add_point_by_click)
        self.plot_canvas.mpl_connect('pick_event', self.on_point_pick)

        # --- Controls Panel ---
        controls_widget = QWidget()
        controls_widget.setFixedWidth(450)  # Increased width for new controls
        controls_layout = QVBoxLayout(controls_widget)
        
        # --- Geometry Input Tabs ---
        self.geom_tabs = QTabWidget()
        self.geom_tabs.currentChanged.connect(self.on_tab_change)

        # Create Outer Shape Tab
        self.outer_tab, self.outer_point_controls = self.create_geometry_tab("Outer Shape")
        self.geom_tabs.addTab(self.outer_tab, "Outer Shape")

        # Create Inner Shape Tab
        self.inner_tab, self.inner_point_controls = self.create_geometry_tab("Inner Hole")
        self.geom_tabs.addTab(self.inner_tab, "Inner Hole")
        
        controls_layout.addWidget(self.geom_tabs)

        # --- Analysis and Plotting Controls ---
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("<b>Analysis Controls</b>"))
        
        self.mesh_size_input = QLineEdit("10.0") # Default to 10mm
        self.generate_mesh_button = QPushButton("Generate Mesh & Solve")
        self.generate_mesh_button.clicked.connect(self.run_analysis)
        controls_layout.addWidget(QLabel("Mesh Size [mm]:"))
        controls_layout.addWidget(self.mesh_size_input)
        controls_layout.addWidget(self.generate_mesh_button)

        # --- Material and Load Inputs ---
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("<b>Inputs</b>"))
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Shear Modulus (G) [MPa]:"))
        self.g_input = QLineEdit("80e3") # Default to steel (e.g., 80 GPa = 80e3 MPa)
        input_layout.addWidget(self.g_input)
        controls_layout.addLayout(input_layout)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Applied Torque (T) [N-m]:"))
        self.t_input = QLineEdit("1000") # Default to 1000 Nm
        input_layout.addWidget(self.t_input)
        controls_layout.addLayout(input_layout)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Beam Length (L) [m]:"))
        self.l_input = QLineEdit("2.0") # Default to 2.0 m
        input_layout.addWidget(self.l_input)
        controls_layout.addLayout(input_layout)

        controls_layout.addStretch()

        # --- Results Display ---
        controls_layout.addSpacing(20)
        self.results_label = QLabel("<b>Results:</b>")
        self.torsional_constant_label = QLabel("Torsional Constant (J): Not calculated")
        self.stiffness_label = QLabel("Torsional Stiffness (k): Not calculated")
        self.twist_angle_label = QLabel("Angle of Twist (θ): Not calculated")
        self.max_stress_label = QLabel("Max Shear Stress (τ_max): Not calculated")
        controls_layout.addWidget(self.results_label)
        controls_layout.addWidget(self.torsional_constant_label)
        controls_layout.addWidget(self.stiffness_label)
        controls_layout.addWidget(self.twist_angle_label)
        controls_layout.addWidget(self.max_stress_label)

        main_layout.addWidget(controls_widget)
        main_layout.addWidget(self.plot_canvas)

        # Set initial state
        self.on_tab_change(0)

    def create_menu_bar(self):
        """Creates the menu bar with File and View menu options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # New project action
        new_action = QAction('&New Project', self)
        new_action.setShortcut('Ctrl+N')
        new_action.setStatusTip('Create a new project')
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        file_menu.addSeparator()
        
        # Open action
        open_action = QAction('&Open...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open a saved project')
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        # Save action
        save_action = QAction('&Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save current project')
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        # Save As action
        save_as_action = QAction('Save &As...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.setStatusTip('Save current project with new name')
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Export results action
        export_action = QAction('&Export Results...', self)
        export_action.setStatusTip('Export analysis results to file')
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Reset zoom action
        reset_zoom_action = QAction('&Reset Zoom', self)
        reset_zoom_action.setShortcut('Ctrl+0')
        reset_zoom_action.setStatusTip('Reset plot zoom to fit all geometry')
        reset_zoom_action.triggered.connect(self.reset_zoom)
        view_menu.addAction(reset_zoom_action)
        
        view_menu.addSeparator()
        
        # Plot instructions action
        plot_help_action = QAction('&Plot Controls Help', self)
        plot_help_action.setStatusTip('Show help for plot navigation controls')
        plot_help_action.triggered.connect(self.show_plot_help)
        view_menu.addAction(plot_help_action)

    def create_geometry_tab(self, name):
        """Creates a widget and its controls for a geometry tab."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # Point List and Editing Section
        points_group = QGroupBox("Points List & Editing")
        points_layout = QVBoxLayout(points_group)
        
        # Points list widget (replaces QTextEdit)
        points_list = QListWidget()
        points_list.setSelectionMode(QListWidget.SingleSelection)
        points_layout.addWidget(points_list)
        
        # Point editing controls
        edit_layout = QHBoxLayout()
        edit_x_input = QDoubleSpinBox()
        edit_x_input.setRange(-99999, 99999)
        edit_x_input.setDecimals(4)
        edit_x_input.setSuffix(" mm")
        edit_y_input = QDoubleSpinBox()
        edit_y_input.setRange(-99999, 99999)
        edit_y_input.setDecimals(4)
        edit_y_input.setSuffix(" mm")
        update_point_button = QPushButton("Update Point")
        update_point_button.setEnabled(False)
        
        edit_layout.addWidget(QLabel("X:"))
        edit_layout.addWidget(edit_x_input)
        edit_layout.addWidget(QLabel("Y:"))
        edit_layout.addWidget(edit_y_input)
        edit_layout.addWidget(update_point_button)
        points_layout.addLayout(edit_layout)
        
        # Point reordering controls
        reorder_layout = QHBoxLayout()
        move_up_button = QPushButton("Move Up")
        move_down_button = QPushButton("Move Down")
        delete_point_button = QPushButton("Delete Point")
        move_up_button.setEnabled(False)
        move_down_button.setEnabled(False)
        delete_point_button.setEnabled(False)
        
        reorder_layout.addWidget(move_up_button)
        reorder_layout.addWidget(move_down_button)
        reorder_layout.addWidget(delete_point_button)
        points_layout.addLayout(reorder_layout)
        
        layout.addWidget(points_group)

        # Action Buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        
        button_layout = QHBoxLayout()
        undo_button = QPushButton("Undo Last")
        clear_button = QPushButton("Clear Geometry")
        button_layout.addWidget(undo_button)
        button_layout.addWidget(clear_button)
        action_layout.addLayout(button_layout)
        layout.addWidget(action_group)

        # Bulk Import Section
        paste_group = QGroupBox(f"Add Points for {name}")
        paste_layout = QVBoxLayout(paste_group)
        paste_layout.addWidget(QLabel("Enter Points (x,y per line or single x,y):"))
        paste_area = QTextEdit()
        paste_area.setFixedHeight(100)
        paste_area.setPlaceholderText("Examples:\n10, 20\n30 40\n(50, 60)")
        paste_button = QPushButton("Add Points")
        paste_layout.addWidget(paste_area)
        paste_layout.addWidget(paste_button)
        layout.addWidget(paste_group)

        # Store controls in a dictionary
        controls = {
            "points_list": points_list, "undo_button": undo_button,
            "clear_button": clear_button, "paste_area": paste_area,
            "paste_button": paste_button, "edit_x_input": edit_x_input,
            "edit_y_input": edit_y_input, "update_point_button": update_point_button,
            "move_up_button": move_up_button, "move_down_button": move_down_button,
            "delete_point_button": delete_point_button
        }

        # Connect signals
        undo_button.clicked.connect(lambda: self.undo_last_point())
        clear_button.clicked.connect(lambda: self.clear_geometry())
        paste_button.clicked.connect(lambda: self.add_points_from_paste())
        points_list.itemSelectionChanged.connect(lambda: self.on_point_selection_changed())
        update_point_button.clicked.connect(lambda: self.update_selected_point())
        move_up_button.clicked.connect(lambda: self.move_point_up())
        move_down_button.clicked.connect(lambda: self.move_point_down())
        delete_point_button.clicked.connect(lambda: self.delete_selected_point())

        return tab_widget, controls

    def on_tab_change(self, index):
        """Handle switching between the geometry tabs."""
        if index == 0:
            self.active_points_list = self.outer_points
            log.info("Switched to Outer Shape tab.")
        else:
            self.active_points_list = self.inner_points
            log.info("Switched to Inner Hole tab.")
        
        # Reset selection when switching tabs
        self.selected_point_index = -1
        self.update_point_display()
        self.update_plot()

    def get_active_controls(self):
        """Returns the controls dictionary for the currently selected tab."""
        if self.geom_tabs.currentIndex() == 0:
            return self.outer_point_controls
        else:
            return self.inner_point_controls

    def add_point(self, x, y):
        """Adds a point to the active list and updates the plot."""
        self.active_points_list.append((x, y))
        log.info(f"Point added: ({x}, {y}). Total points in active list: {len(self.active_points_list)}")
        self.update_point_display()
        self.update_plot()
        self._mark_as_modified()

    def add_point_by_click(self, event):
        """Adds a point to the list based on a click on the plot."""
        if event.inaxes != self.plot_canvas.axes:
            return
        if event.button != 1:  # Left mouse button
            return
        x, y = event.xdata, event.ydata
        log.debug(f"Canvas click detected at raw coordinates: ({x}, {y})")
        self.add_point(round(x, 4), round(y, 4))

    def undo_last_point(self):
        """Removes the last added point from the active list."""
        if self.active_points_list:
            removed_point = self.active_points_list.pop()
            log.info(f"Undid last point: {removed_point}. Points remaining: {len(self.active_points_list)}")
            self.update_point_display()
            self.update_plot()
            self._mark_as_modified()
        else:
            log.warning("Undo attempted with no points in the list.")

    def clear_geometry(self):
        """Clears all points from the active list and resets the plot."""
        log.info(f"Clearing geometry. Removed {len(self.active_points_list)} points from active list.")
        self.active_points_list.clear()
        if self.geom_tabs.currentIndex() == 0: # If clearing outer, also clear results
             self.reset_results_labels()
        self.update_point_display()
        self.update_plot()
        self._mark_as_modified()

    def reset_results_labels(self):
        """Resets the text of the result labels."""
        self.torsional_constant_label.setText("Torsional Constant (J): Not calculated")
        self.stiffness_label.setText("Torsional Stiffness (k): Not calculated")
        self.twist_angle_label.setText("Angle of Twist (θ): Not calculated")
        self.max_stress_label.setText("Max Shear Stress (τ_max): Not calculated")
        log.debug("Results labels reset.")

    def update_point_display(self):
        """Updates the QListWidget for the active tab with the current list of points."""
        controls = self.get_active_controls()
        points_list = controls["points_list"]
        points_list.clear()
        
        for i, (x, y) in enumerate(self.active_points_list):
            item_text = f"Point {i+1}: ({x}, {y})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)  # Store the index
            points_list.addItem(item)

    def on_point_selection_changed(self):
        """Handles selection changes in the points list."""
        controls = self.get_active_controls()
        points_list = controls["points_list"]
        selected_items = points_list.selectedItems()
        
        if selected_items:
            item = selected_items[0]
            self.selected_point_index = item.data(Qt.UserRole)
            x, y = self.active_points_list[self.selected_point_index]
            
            # Update edit controls
            controls["edit_x_input"].setValue(x)
            controls["edit_y_input"].setValue(y)
            
            # Enable editing buttons
            controls["update_point_button"].setEnabled(True)
            controls["move_up_button"].setEnabled(self.selected_point_index > 0)
            controls["move_down_button"].setEnabled(self.selected_point_index < len(self.active_points_list) - 1)
            controls["delete_point_button"].setEnabled(True)
            
            # Highlight the selected point in the plot
            self.update_plot()
        else:
            self.selected_point_index = -1
            # Disable editing buttons
            controls["update_point_button"].setEnabled(False)
            controls["move_up_button"].setEnabled(False)
            controls["move_down_button"].setEnabled(False)
            controls["delete_point_button"].setEnabled(False)

    def update_selected_point(self):
        """Updates the coordinates of the selected point."""
        if self.selected_point_index >= 0:
            controls = self.get_active_controls()
            new_x = controls["edit_x_input"].value()
            new_y = controls["edit_y_input"].value()
            
            old_point = self.active_points_list[self.selected_point_index]
            self.active_points_list[self.selected_point_index] = (new_x, new_y)
            
            log.info(f"Point {self.selected_point_index + 1} updated from {old_point} to ({new_x}, {new_y})")
            
            self.update_point_display()
            self.update_plot()
            self._mark_as_modified()

    def move_point_up(self):
        """Moves the selected point up in the list (earlier in the order)."""
        if self.selected_point_index > 0:
            # Swap with previous point
            idx = self.selected_point_index
            self.active_points_list[idx], self.active_points_list[idx-1] = \
                self.active_points_list[idx-1], self.active_points_list[idx]
            
            # Update selection
            self.selected_point_index = idx - 1
            
            log.info(f"Point moved up to position {self.selected_point_index + 1}")
            
            self.update_point_display()
            self.update_plot()
            self._mark_as_modified()
            
            # Reselect the moved point
            controls = self.get_active_controls()
            points_list = controls["points_list"]
            points_list.setCurrentRow(self.selected_point_index)

    def move_point_down(self):
        """Moves the selected point down in the list (later in the order)."""
        if self.selected_point_index < len(self.active_points_list) - 1:
            # Swap with next point
            idx = self.selected_point_index
            self.active_points_list[idx], self.active_points_list[idx+1] = \
                self.active_points_list[idx+1], self.active_points_list[idx]
            
            # Update selection
            self.selected_point_index = idx + 1
            
            log.info(f"Point moved down to position {self.selected_point_index + 1}")
            
            self.update_point_display()
            self.update_plot()
            self._mark_as_modified()
            
            # Reselect the moved point
            controls = self.get_active_controls()
            points_list = controls["points_list"]
            points_list.setCurrentRow(self.selected_point_index)

    def delete_selected_point(self):
        """Deletes the selected point from the list."""
        if self.selected_point_index >= 0:
            deleted_point = self.active_points_list.pop(self.selected_point_index)
            log.info(f"Deleted point {self.selected_point_index + 1}: {deleted_point}")
            
            # Adjust selection
            if self.selected_point_index >= len(self.active_points_list):
                self.selected_point_index = len(self.active_points_list) - 1
            
            self.update_point_display()
            self.update_plot()
            self._mark_as_modified()
            
            # Clear selection if no points left
            if not self.active_points_list:
                self.selected_point_index = -1

    def on_point_pick(self, event):
        """Handles clicking on points in the plot for selection."""
        if hasattr(event, 'ind') and event.ind is not None:
            # Get the index of the clicked point
            point_idx = event.ind[0]
            if 0 <= point_idx < len(self.active_points_list):
                controls = self.get_active_controls()
                points_list = controls["points_list"]
                points_list.setCurrentRow(point_idx)
                log.debug(f"Point {point_idx + 1} selected via plot click")

    def update_plot(self, phi=None, V=None):
        """Redraws the geometry or plots the stress function contour."""
        self.plot_canvas.axes.clear()
        if phi is not None and V is not None:
            self.plot_canvas.plot_contour(phi, V)
        else:
            # Determine which points list is active
            active_is_outer = (self.geom_tabs.currentIndex() == 0)
            self.plot_canvas.plot_polygons(self.outer_points, self.inner_points, 
                                         self.selected_point_index, active_is_outer)
        log.debug("Plot updated.")

    def add_points_from_paste(self):
        """Parses text from the paste area of the active tab and adds the points."""
        controls = self.get_active_controls()
        paste_area = controls["paste_area"]
        text = paste_area.toPlainText()
        lines = text.split('\n')
        added_count = 0
        error_lines = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                # Try to parse various formats like "x,y", "x y", "(x, y)"
                clean_line = line.replace('(', '').replace(')', '').replace(',', ' ')
                parts = clean_line.split()
                if len(parts) == 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    self.add_point(x, y)
                    added_count += 1
                else:
                    error_lines.append(f"Line {i+1}: '{line}'")
            except (ValueError, IndexError):
                error_lines.append(f"Line {i+1}: '{line}'")
        
        log.info(f"Pasted points processed. Added: {added_count}, Errors: {len(error_lines)}")
        if error_lines:
            log.warning(f"Could not parse pasted lines: {error_lines}")
            QMessageBox.warning(self, "Parsing Error",
                                f"Successfully added {added_count} points.\n"
                                f"Could not parse the following lines:\n" +
                                "\n".join(error_lines))
        
        paste_area.clear()
        self.update_point_display()

    def run_analysis(self):
        """
        Runs the full meshing and solving process.
        """
        log.info("Starting analysis run.")
        if len(self.outer_points) < 3:
            log.warning("Analysis attempted with insufficient points for outer shape.")
            QMessageBox.warning(self, "Geometry Error", "Please define at least 3 points for the Outer Shape.")
            return
        
        if len(self.inner_points) > 0 and len(self.inner_points) < 3:
            log.warning("Analysis attempted with insufficient points for inner hole.")
            QMessageBox.warning(self, "Geometry Error", "If you define an Inner Hole, it must have at least 3 points.")
            return

        try:
            # --- Unit Conversions ---
            outer_points_m = [(p[0] / 1000.0, p[1] / 1000.0) for p in self.outer_points]
            inner_points_m = [(p[0] / 1000.0, p[1] / 1000.0) for p in self.inner_points]
            mesh_size_m = float(self.mesh_size_input.text()) / 1000.0
            
            G_Pa = float(self.g_input.text()) * 1e6
            T_Nm = float(self.t_input.text())
            L_beam_m = float(self.l_input.text())
            log.info(f"Inputs converted for solver: G={G_Pa:.2e} Pa, T={T_Nm} Nm, L={L_beam_m} m, mesh_size={mesh_size_m} m")

        except ValueError:
            log.error("Invalid numerical input for analysis.", exc_info=True)
            QMessageBox.warning(self, "Input Error", "Please ensure Mesh Size, G, T, and L are valid numbers.")
            return

        # --- Progress Dialog ---
        progress = QProgressDialog("Running analysis...", "Cancel", 0, 4, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0) # Show immediately
        
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # 1. Meshing
            progress.setValue(1)
            progress.setLabelText("Meshing cross-section...")
            log.info("Starting mesh generation.")
            QApplication.processEvents()
            if progress.wasCanceled():
                raise InterruptedError("Analysis cancelled by user.")

            mesh_dir = meshing.create_mesh(outer_points_m, inner_points_m, mesh_size_m)
            log.info(f"Meshing complete. Mesh files in: {mesh_dir}")

            # 2. Solving
            progress.setValue(2)
            progress.setLabelText("Solving FEA problem...")
            log.info("Starting FEA solver.")
            QApplication.processEvents()
            if progress.wasCanceled():
                raise InterruptedError("Analysis cancelled by user.")

            J, k, theta, tau_max, phi, V = solver.solve_torsion(mesh_dir, G_Pa, T_Nm, L_beam_m)
            log.info(f"Solver finished. J={J:.4e}, k={k:.4e}, theta={theta:.4e}, tau_max={tau_max:.4e}")

            # 3. Updating GUI
            progress.setValue(3)
            progress.setLabelText("Updating results...")
            log.info("Updating GUI with results.")
            QApplication.processEvents()

            # Update result labels (with conversions for display)
            J_mm4 = J * (1000**4) # m^4 to mm^4
            self.results_label.setText("<b>Results:</b>")
            self.torsional_constant_label.setText(f"Torsional Constant (J): {J_mm4:.4e} mm^4")
            self.stiffness_label.setText(f"Torsional Stiffness (k): {k:.4e} Nm/rad")
            self.twist_angle_label.setText(f"Angle of Twist (θ): {np.rad2deg(theta):.4f} degrees")
            self.max_stress_label.setText(f"Max Shear Stress (τ_max): {tau_max/1e6:.2f} MPa")

            # 4. Update plot with contour of phi
            self.update_plot(phi, V)
            
            progress.setValue(4) # Finish
            log.info("Analysis run completed successfully.")

        except InterruptedError as e:
             log.warning(f"Analysis was cancelled by the user.")
             self.results_label.setText(f"<b>Results:</b> {e}")
        except Exception as e:
            log.critical("An unhandled exception occurred during analysis.", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()

    def new_project(self):
        """Creates a new project by clearing all data."""
        reply = QMessageBox.question(self, 'New Project', 
                                   'Are you sure you want to create a new project?\nAll unsaved changes will be lost.',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.outer_points.clear()
            self.inner_points.clear()
            self.current_file_path = None
            self.setWindowTitle("2D Torsional Analysis FEA - New Project")
            
            # Reset all input fields
            self.mesh_size_input.setText("10.0")
            self.g_input.setText("80e3")
            self.t_input.setText("1000")
            self.l_input.setText("2.0")
            
            # Reset results
            self.reset_results_labels()
            
            # Update displays
            self.update_point_display()
            self.update_plot()
            
            log.info("New project created")

    def save_project(self):
        """Saves the current project."""
        if self.current_file_path:
            self._save_to_file(self.current_file_path)
        else:
            self.save_project_as()

    def save_project_as(self):
        """Saves the current project with a new filename."""
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Save Project As", 
            "project.fea",  # Default filename with .fea extension
            "FEA Project Files (*.fea);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            # Ensure the file has the correct extension based on the selected filter
            if selected_filter.startswith("FEA Project Files") and not file_path.endswith('.fea'):
                file_path += '.fea'
            elif selected_filter.startswith("JSON Files") and not file_path.endswith('.json'):
                file_path += '.json'
            
            self._save_to_file(file_path)

    def _save_to_file(self, file_path):
        """Internal method to save project data to a file."""
        try:
            project_data = {
                "version": "1.0",
                "geometry": {
                    "outer_points": self.outer_points,
                    "inner_points": self.inner_points
                },
                "parameters": {
                    "mesh_size": self.mesh_size_input.text(),
                    "shear_modulus": self.g_input.text(),
                    "applied_torque": self.t_input.text(),
                    "beam_length": self.l_input.text()
                },
                "results": {
                    "torsional_constant": self.torsional_constant_label.text(),
                    "stiffness": self.stiffness_label.text(),
                    "twist_angle": self.twist_angle_label.text(),
                    "max_stress": self.max_stress_label.text()
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            self.current_file_path = file_path
            self.setWindowTitle(f"2D Torsional Analysis FEA - {os.path.basename(file_path)}")
            
            QMessageBox.information(self, "Save Successful", f"Project saved to:\n{file_path}")
            log.info(f"Project saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{str(e)}")
            log.error(f"Failed to save project: {e}")

    def open_project(self):
        """Opens a saved project file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Project", 
            "",  # Start in current directory
            "FEA Project Files (*.fea);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self._load_from_file(file_path)

    def _load_from_file(self, file_path):
        """Internal method to load project data from a file."""
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)
            
            # Load geometry
            geometry = project_data.get("geometry", {})
            self.outer_points = geometry.get("outer_points", [])
            self.inner_points = geometry.get("inner_points", [])
            
            # Load parameters
            parameters = project_data.get("parameters", {})
            self.mesh_size_input.setText(parameters.get("mesh_size", "10.0"))
            self.g_input.setText(parameters.get("shear_modulus", "80e3"))
            self.t_input.setText(parameters.get("applied_torque", "1000"))
            self.l_input.setText(parameters.get("beam_length", "2.0"))
            
            # Load results if available
            results = project_data.get("results", {})
            if results:
                self.torsional_constant_label.setText(results.get("torsional_constant", "Torsional Constant (J): Not calculated"))
                self.stiffness_label.setText(results.get("stiffness", "Torsional Stiffness (k): Not calculated"))
                self.twist_angle_label.setText(results.get("twist_angle", "Angle of Twist (θ): Not calculated"))
                self.max_stress_label.setText(results.get("max_stress", "Max Shear Stress (τ_max): Not calculated"))
            
            # Update active points list based on current tab
            self.active_points_list = self.outer_points if self.geom_tabs.currentIndex() == 0 else self.inner_points
            
            # Update displays
            self.update_point_display()
            self.update_plot()
            
            self.current_file_path = file_path
            self.setWindowTitle(f"2D Torsional Analysis FEA - {os.path.basename(file_path)}")
            
            QMessageBox.information(self, "Load Successful", f"Project loaded from:\n{file_path}")
            log.info(f"Project loaded from {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{str(e)}")
            log.error(f"Failed to load project: {e}")

    def export_results(self):
        """Exports analysis results to a text file."""
        if "Not calculated" in self.torsional_constant_label.text():
            QMessageBox.warning(self, "No Results", "No analysis results to export. Please run an analysis first.")
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Export Results", 
            "results.txt",  # Default filename
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            # Ensure the file has the correct extension based on the selected filter
            if selected_filter.startswith("Text Files") and not file_path.endswith('.txt'):
                file_path += '.txt'
            elif selected_filter.startswith("CSV Files") and not file_path.endswith('.csv'):
                file_path += '.csv'
            
            try:
                with open(file_path, 'w') as f:
                    f.write("2D Torsional Analysis FEA - Results Export\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Analysis Parameters:\n")
                    f.write(f"- Mesh Size: {self.mesh_size_input.text()} mm\n")
                    f.write(f"- Shear Modulus (G): {self.g_input.text()} MPa\n")
                    f.write(f"- Applied Torque (T): {self.t_input.text()} N-m\n")
                    f.write(f"- Beam Length (L): {self.l_input.text()} m\n\n")
                    f.write(f"Geometry:\n")
                    f.write(f"- Outer Points ({len(self.outer_points)}): {self.outer_points}\n")
                    f.write(f"- Inner Points ({len(self.inner_points)}): {self.inner_points}\n\n")
                    f.write(f"Results:\n")
                    f.write(f"- {self.torsional_constant_label.text()}\n")
                    f.write(f"- {self.stiffness_label.text()}\n")
                    f.write(f"- {self.twist_angle_label.text()}\n")
                    f.write(f"- {self.max_stress_label.text()}\n")
                
                QMessageBox.information(self, "Export Successful", f"Results exported to:\n{file_path}")
                log.info(f"Results exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
                log.error(f"Failed to export results: {e}")

    def reset_zoom(self):
        """Resets the plot zoom to fit all geometry."""
        self.plot_canvas.reset_view()
        log.debug("Plot zoom reset to fit all geometry")

    def show_plot_help(self):
        """Shows help dialog for plot navigation controls."""
        help_text = """
Plot Navigation Controls:

🖱️ Mouse Controls:
• Mouse Wheel: Zoom in/out centered on cursor
• Right-click + Drag: Pan around the plot
• Left-click: Add new point to active geometry
• Click on Point: Select point for editing

⌨️ Keyboard Shortcuts:
• Ctrl+0: Reset zoom to fit all geometry

✏️ Point Editing:
• Select points from the list or by clicking
• Use the edit controls to modify coordinates
• Use Move Up/Down to reorder points
• Delete points with the Delete button

📐 Tips:
• Point order determines the shape boundary
• Use zoom to precisely position points
• Right-click drag to pan when zoomed in
"""
        QMessageBox.information(self, "Plot Controls Help", help_text)

    def _mark_as_modified(self):
        """Marks the project as modified by adding an asterisk to the title."""
        current_title = self.windowTitle()
        if not current_title.endswith("*"):
            self.setWindowTitle(current_title + "*")


if __name__ == '__main__':
    # This allows you to run the GUI directly for testing
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

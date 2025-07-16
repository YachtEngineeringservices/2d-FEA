"""
Simplified Windows version of the 2D FEA application.
This version includes the GUI and basic functionality but uses
simplified solver due to DOLFINx Windows installation complexity.
"""

import sys
import logging
from PySide6.QtWidgets import QApplication, QMessageBox

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fea_app_windows.log'),
        logging.StreamHandler()
    ]
)

def show_platform_info():
    """Show information about Windows compatibility"""
    msg = QMessageBox()
    msg.setWindowTitle("2D FEA - Windows Edition")
    msg.setIcon(QMessageBox.Information)
    msg.setText("Welcome to 2D FEA Torsion Analysis - Windows Edition")
    msg.setInformativeText(
        "This Windows version includes:\n"
        "✓ Full GUI interface\n"
        "✓ Geometry input and visualization\n"
        "✓ Mesh generation (GMSH)\n"
        "✓ Save/Load project files\n"
        "✓ Results visualization\n\n"
        "⚠ Note: Advanced FEA solving requires DOLFINx\n"
        "For full FEA functionality, use Linux/WSL2 version\n"
        "or install DOLFINx separately on Windows."
    )
    msg.setDetailedText(
        "DOLFINx Installation on Windows:\n"
        "1. Install conda/mamba\n"
        "2. conda install -c conda-forge fenics-dolfinx\n"
        "3. Replace this simplified solver with full version"
    )
    msg.exec()

def main():
    """Main application entry point"""
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("2D FEA Torsion Analysis")
        app.setApplicationVersion("1.0 Windows Edition")
        
        # Show platform information
        show_platform_info()
        
        # Import and create main window
        from gui.main_window import MainWindow
        window = MainWindow()
        
        # Show the main window
        window.show()
        
        logging.info("Application started successfully")
        
        # Run the application
        return app.exec()
        
    except ImportError as e:
        # Handle missing dependencies
        error_msg = f"Missing dependency: {e}\n\nPlease install required packages:\npip install PySide6 matplotlib numpy gmsh meshio h5py"
        
        app = QApplication(sys.argv)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Missing Dependencies")
        msg.setText("Required Python packages are missing")
        msg.setInformativeText(error_msg)
        msg.exec()
        
        logging.error(f"Import error: {e}")
        return 1
        
    except Exception as e:
        # Handle other errors
        logging.error(f"Application error: {e}")
        
        app = QApplication(sys.argv)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Application Error")
        msg.setText(f"An error occurred: {e}")
        msg.exec()
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

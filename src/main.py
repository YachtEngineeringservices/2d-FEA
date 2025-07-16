import sys
import logging
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def setup_logging():
    """Configures logging to a file and the console."""
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler("fea_app.log", mode='w')
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main():
    """
    The main entry point for the 2D FEA application.
    """
    setup_logging()
    logging.info("Application starting.")

    # Create the Qt application
    app = QApplication(sys.argv)

    # Create and show the main window
    window = MainWindow()
    window.show()

    # Start the Qt event loop
    try:
        sys.exit(app.exec())
    except Exception as e:
        logging.critical("Unhandled exception caught, application terminating.", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

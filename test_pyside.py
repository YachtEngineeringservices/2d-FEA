#!/usr/bin/env python
import sys
from PySide6.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)
label = QLabel("Hello PySide6!")
label.show()
print("PySide6 test successful")
sys.exit(0)  # Exit immediately for testing

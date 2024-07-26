import os
import sys

from ultrasound_segmentation import UltrasoundApp
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)

# Global variable to hold the Holoscan application instance
gApp = None


# Worker class to run the Holoscan application in a separate thread
class UltrasoundWorker(QObject):
    finished = Signal()  # Signal to indicate the worker has finished
    progress = Signal(int)  # Signal to indicate progress (if needed)

    def run(self):
        """Run the Holoscan application."""
        config_file = os.path.join(os.path.dirname(__file__), "ultrasound_segmentation.yaml")
        data_file = "/workspace/holohub/data/ultrasound_segmentation"
        global gApp
        gApp = app = UltrasoundApp(source="replayer", data=data_file)
        app.config(config_file)
        app.run()


# Main window class for the PySide2 UI
class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runHoloscanApp()  # Run the Holoscan application

    def runHoloscanApp(self):
        """Run the Holoscan application in a separate thread."""
        self.thread = QThread()
        self.worker = UltrasoundWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            self.thread.quit()
            self.thread.wait()
            self.close()
        if event.key() == Qt.Key_Space:
            # Set parameters in the Holoscan application
            global gApp
            if gApp:
                gApp.set_overlay_op_parameters()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

from PyQt5 import QtCore, QtGui, QtWidgets
from dataProcessingDialog import DataProcessingDialog

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = DataProcessingDialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

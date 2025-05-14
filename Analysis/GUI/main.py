
import sys
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication
# from PyQt6.uic import loadUi

from Analysis.GUI.elav_gui import Eval_gui
from Analysis.GUI.build_gui import Build_gui
from hog_gui import HOG_gui

app=QApplication(sys.argv)

widget=QtWidgets.QStackedWidget()
evalwindow=Eval_gui(widget)
buildwindow=Build_gui(widget)
hogwindow=HOG_gui(widget)
widget.addWidget(evalwindow)
widget.addWidget(buildwindow)
widget.addWidget(hogwindow)
widget.setFixedWidth(800)
widget.setFixedHeight(600)
widget.show()
sys.exit(app.exec())
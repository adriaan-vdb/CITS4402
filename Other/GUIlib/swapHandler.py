# Handles Swapping between pages of the GUI
#GUI import
from PyQt6.QtGui import QAction
#File Location
import sys
sys.path.insert(1,"./CITS4401/Other/GUIlib")

class swapHandler():
    def __init__(self, widget,window):
        def toHOG(self):
           widget.setCurrentIndex(2)
        def toBuild(self):
            widget.setCurrentIndex(1)
        def toEval(self):
            widget.setCurrentIndex(0)
        HOG_action = QAction('Modify HOG', window)
        HOG_action.triggered.connect(toHOG)
        Build_action = QAction('Build model', window)
        Build_action.triggered.connect(toBuild)
        Eval_action = QAction('Evaluate Model', window)
        Eval_action.triggered.connect(toEval)
        
        
        window.Select.addAction(HOG_action)
        window.Select.addAction(Build_action)
        window.Select.addAction(Eval_action)

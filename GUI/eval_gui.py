from PyQt6.uic import loadUi
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QPixmap

from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import numpy as np
import openpyxl
from openpyxl.drawing.image import Image

import sys
sys.path.insert(1,"./CITS4401/GUI")
from GUI.swapHandler import swapHandler



class Eval_gui(QMainWindow):


    def __init__(self,widget):
        super(Eval_gui,self).__init__()
        loadUi("GUI/EvaluateWindow.ui",self)
        self.Model_Import.clicked.connect(self.import_M)
        self.Model_Evaluate.clicked.connect(self.evaluate_M)
        self.Select_Test_Set.setPlainText("Analysis/data/raw")
        self.Select_Model.setPlainText("Analysis/svm_model.pkl")
        self.swap=swapHandler(widget,self)

        # exit_action = QAction('Exit', self)
        # exit_action.triggered.connect(self.toHOG)

        # # self.menuHOGModify.clicked.connect(self.toHOG)
        # self.Select.addAction(exit_action)
        # # self.menuModelBuild.clicked.connect(self.toBuild)
        # # self.menuevaluate.clicked.connect(self.toEval)

        self.nav_eval = QtWidgets.QPushButton("Go to Evaluate", self)
        self.nav_eval.setGeometry(10, 500, 120, 30)
        self.nav_eval.clicked.connect(lambda: widget.setCurrentIndex(0))

        self.nav_build = QtWidgets.QPushButton("Go to Build", self)
        self.nav_build.setGeometry(140, 500, 120, 30)
        self.nav_build.clicked.connect(lambda: widget.setCurrentIndex(1))

        self.nav_hog = QtWidgets.QPushButton("Go to HOG", self)
        self.nav_hog.setGeometry(270, 500, 120, 30)
        self.nav_hog.clicked.connect(lambda: widget.setCurrentIndex(2))

    
    def import_M(self):
        self.Model_Import.setEnabled(False)
        with open(self.Select_Model.toPlainText(), 'rb') as f:
            self.model = pickle.load(f)
        
        self.testdata=[]
        self.trueval=[]
        self.path=[]
        Tnum=0
        Fnum=0
        for entry in os.scandir(self.Select_Test_Set.toPlainText()):
            if entry.name.endswith("1.txt") and Tnum<100:
                self.testdata.append(np.loadtxt(entry.path, delimiter=',').flatten())
                Tnum=Tnum+1 
                self.trueval.append(1)
                self.path.append("Analysis/data/processed/human/"+entry.name[:-5])
            elif entry.name.endswith("0.txt") and Fnum<100:
                self.testdata.append(np.loadtxt(entry.path, delimiter=',').flatten())
                Fnum=Fnum+1
                self.trueval.append(0)
                self.path.append("Analysis/data/processed/nonhuman/"+entry.name[:-5])
        
        self.Model_Evaluate.setEnabled(True)
        self.Model_Import.setEnabled(True)

        print("Import Successful")
        print(self.Select_Test_Set.toPlainText())
        print(self.Select_Model.toPlainText())
        print(self.path)
    
    def evaluate_M(self):
        self.Model_Evaluate.setEnabled(False)
        print("seems good")
        self.q=self.model.predict(self.testdata)
        # score=len(self.trueval)
        # for guess in range(0,score):
        #     score=score-(self.q[guess]+self.trueval[guess])%2
        print("Test set evaluation:")
        print(repr(classification_report(self.trueval, self.q)))
        print(repr(accuracy_score(self.trueval, self.q)))
        
        self.PRE_T.setText("Expected: "+str(self.trueval[0]))
        self.PRE_E.setText("Predicted: "+str(self.q[0]))
        self.Acc.setText("Accuracy: "+str(accuracy_score(self.trueval, self.q)*100)+"%")
        self.Report.setPlainText(classification_report(self.trueval, self.q))
        pix=QPixmap(self.path[0])
        self.Image.setPixmap(pix)
        self.Image.show()
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet["A1"]="Predicted"
        sheet["B1"]="Actual"
        sheet["C1"]="Image"
        sheet["D1"]="Path"
        sheet.row_dimensions[1].height = 100
        for i in range(len(self.q)):
            sheet.row_dimensions[i+2].height = 100
            sheet[f"A{i+2}"]=self.q[i]
            sheet[f"B{i+2}"]=self.trueval[i]
            img = Image(self.path[i])
            sheet.add_image(img,f"C{i+2}")
            sheet[f"D{i+2}"]=self.path[i]
        workbook.save('output.xlsx')

        self.Model_Evaluate.setEnabled(True)
        self.Image_select.setEnabled(True)
        self.Image_select.setRange(0,len(self.testdata)-1)
        self.Image_select.valueChanged.connect(self.swapImage)

    def swapImage(self,value):
        self.PRE_T.setText("Expected: "+str(self.trueval[value]))
        self.PRE_E.setText("Predicted: "+str(self.q[value]))
        pix=QPixmap(self.path[value])
        self.Image.setPixmap(pix)
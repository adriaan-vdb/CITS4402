
from PyQt6.uic import loadUi
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QPixmap

from sklearn import svm
import pickle
import os
import numpy as np


from swapHandler import swapHandler

class Eval_gui(QMainWindow):


    def __init__(self,widget):
        super(Eval_gui,self).__init__()
        loadUi("EvaluateWindow.ui",self)
        self.Model_Import.clicked.connect(self.import_M)
        self.Model_Evaluate.clicked.connect(self.evaluate_M)
        self.Select_Test_Set.setPlainText("trainHOG")
        self.Select_Model.setPlainText("model.pkl")
        self.swap=swapHandler(widget,self)

        # exit_action = QAction('Exit', self)
        # exit_action.triggered.connect(self.toHOG)

        # # self.menuHOGModify.clicked.connect(self.toHOG)
        # self.Select.addAction(exit_action)
        # # self.menuModelBuild.clicked.connect(self.toBuild)
        # # self.menuevaluate.clicked.connect(self.toEval)


    
    def import_M(self):
        self.Model_Import.setEnabled(False)
        with open(self.Select_Model.toPlainText(), 'rb') as f:
            self.model = pickle.load(f)
        
        self.testdata=[]
        self.results=[]
        self.path=[]
        Tnum=0
        Fnum=0
        for entry in os.scandir(self.Select_Test_Set.toPlainText()):
            if entry.name.endswith("T.txt") and Tnum<100:
                self.testdata.append(np.loadtxt(entry.path, delimiter=',').flatten())
                Tnum=Tnum+1 
                self.results.append(1)
                self.path.append(entry.name[:-5])
            elif entry.name.endswith("F.txt") and Fnum<100:
                self.testdata.append(np.loadtxt(entry.path, delimiter=',').flatten())
                Fnum=Fnum+1
                self.results.append(0)
                self.path.append(entry.name[:-5])
        
        self.Model_Evaluate.setEnabled(True)
        self.Model_Import.setEnabled(True)

        print("Import Successful")
        print(self.Select_Test_Set.toPlainText())
        print(self.Select_Model.toPlainText())
    
    def evaluate_M(self):
        self.Model_Evaluate.setEnabled(False)
        print("seems good")
        self.q=self.model.predict(self.testdata)
        score=len(self.results)
        for guess in range(0,score):
            score=score-(self.q[guess]+self.results[guess])%2
        
        self.PRE_T.setText("Expected: "+str(self.results[0]))
        self.PRE_E.setText("Predicted: "+str(self.q[0]))
        self.Acc.setText("Accuracy: "+str(100*(score/len(self.results)))+"%")
        pix=QPixmap(self.Select_Test_Set.toPlainText()+"/"+self.path[0])
        self.Image.setPixmap(pix)
        # self.Image.show()
        self.Model_Evaluate.setEnabled(True)
        self.Image_select.setEnabled(True)
        self.Image_select.setRange(0,len(self.testdata)-1)
        self.Image_select.valueChanged.connect(self.swapImage)

    def swapImage(self,value):
        self.PRE_T.setText("Expected: "+str(self.results[value]))
        self.PRE_E.setText("Predicted: "+str(self.q[value]))
        pix=QPixmap(self.Select_Test_Set.toPlainText()+"/"+self.path[value])
        self.Image.setPixmap(pix)
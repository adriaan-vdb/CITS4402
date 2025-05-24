# Gui used for training models based on provided HOGs from the HOG Gui


#Model train imports
from sklearn.svm import SVC
import random

#File modification and creation
import joblib
import os
import numpy as np
import sys
sys.path.insert(1,"./CITS4401/Other/GUIlib")

#GUI Imports
from Other.GUIlib.swapHandler import swapHandler
from PyQt6.uic import loadUi
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow




class Build_gui(QMainWindow):


    def __init__(self, widget):
        super(Build_gui,self).__init__()
        loadUi("Other/GUIlib/BuildWindow.ui",self)
        self.Data_Import.clicked.connect(self.import_D)
        self.Model_Fit.clicked.connect(self.fit_M)
        self.save_Model.clicked.connect(self.save_M)
        self.Model_save.setPlainText("model")
        self.Select_Train_set.setPlainText("Other/GUIlib/Output/TrainsetGUI")
        # Inititaiise menu bar
        self.swap=swapHandler(widget,self)

        #start Model
        self.model=SVC(C=10,kernel='rbf')
        # backup buttons in case a mac user does not see the menu bar (it's at the top of the screen)
        self.nav_eval = QtWidgets.QPushButton("Go to Evaluate", self)
        self.nav_eval.setGeometry(10, 500, 120, 30)
        self.nav_eval.clicked.connect(lambda: widget.setCurrentIndex(0))

        self.nav_build = QtWidgets.QPushButton("Go to Build", self)
        self.nav_build.setGeometry(140, 500, 120, 30)
        self.nav_build.clicked.connect(lambda: widget.setCurrentIndex(1))

        self.nav_hog = QtWidgets.QPushButton("Go to HOG", self)
        self.nav_hog.setGeometry(270, 500, 120, 30)
        self.nav_hog.clicked.connect(lambda: widget.setCurrentIndex(2))
    
    # Imports HOG used for training
    def import_D(self):
        self.data=[]
        self.label=[]
        for entry in os.scandir(self.Select_Train_set.toPlainText()):
            if entry.name.endswith("1.txt"):
                self.data.append(np.loadtxt(entry.path, delimiter=',').flatten())
                self.label.append(1)
            if entry.name.endswith("0.txt"):
                self.data.append(np.loadtxt(entry.path, delimiter=',').flatten())
                self.label.append(0)

            
        self.Model_Fit.setEnabled(True)
        self.CValSign.setEnabled(True)
        self.CVal.setEnabled(True)

        print("Import Successful")
    
    #Fits model based on the import
    def fit_M(self):
        self.model=SVC(C=self.CVal.value(),kernel='rbf')
        self.model.fit(self.data,self.label)
        self.save_Model.setEnabled(True)
        self.Model_save.setEnabled(True)

    # Saves model as a .pkl for later use in evaluations 
    def save_M(self):
        with open(self.Model_save.toPlainText()+".pkl",'wb') as f:
            joblib.dump(self.model,f)
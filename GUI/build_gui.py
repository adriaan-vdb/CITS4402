from PyQt6.uic import loadUi
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow

from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# import pickle
import joblib
import os
import numpy as np

import random

import sys
sys.path.insert(1,"./CITS4401/GUI")

from GUI.swapHandler import swapHandler





class Build_gui(QMainWindow):


    def __init__(self, widget):
        super(Build_gui,self).__init__()
        loadUi("GUI/BuildWindow.ui",self)
        self.Data_Import.clicked.connect(self.import_D)
        self.Model_Fit.clicked.connect(self.fit_M)
        self.save_Model.clicked.connect(self.save_M)
        self.Model_save.setPlainText("model")
        self.Select_Train_set.setPlainText("GUI/Output/TrainsetGUI")
        # self.Select_FALSE_set.setPlainText("notpeople")
        self.swap=swapHandler(widget,self)

        ##Model stuff
        self.model=SVC(C=10,kernel='rbf')
    
        self.nav_eval = QtWidgets.QPushButton("Go to Evaluate", self)
        self.nav_eval.setGeometry(10, 500, 120, 30)
        self.nav_eval.clicked.connect(lambda: widget.setCurrentIndex(0))

        self.nav_build = QtWidgets.QPushButton("Go to Build", self)
        self.nav_build.setGeometry(140, 500, 120, 30)
        self.nav_build.clicked.connect(lambda: widget.setCurrentIndex(1))

        self.nav_hog = QtWidgets.QPushButton("Go to HOG", self)
        self.nav_hog.setGeometry(270, 500, 120, 30)
        self.nav_hog.clicked.connect(lambda: widget.setCurrentIndex(2))
    
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
    
    def fit_M(self):
        self.model=SVC(C=self.CVal.value(),kernel='rbf')
        self.model.fit(self.data,self.label)
        self.save_Model.setEnabled(True)
        self.Model_save.setEnabled(True)

    def save_M(self):
        with open(self.Model_save.toPlainText()+".pkl",'wb') as f:
            joblib.dump(self.model,f)
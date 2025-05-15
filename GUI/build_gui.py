from PyQt6.uic import loadUi
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow

from sklearn import svm
import pickle
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
        self.Select_Train_set.setPlainText("TrainsetGUI")
        # self.Select_FALSE_set.setPlainText("notpeople")
        self.swap=swapHandler(widget,self)

        ##Model stuff
        self.model=svm.SVC()
    
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

        print("Import Successful")
    
    def fit_M(self):
        self.model.fit(self.data,self.label)
        self.save_Model.setEnabled(True)
        self.Model_save.setEnabled(True)

    def save_M(self):
        with open(self.Model_save.toPlainText()+".pkl",'wb') as f:
            pickle.dump(self.model,f)
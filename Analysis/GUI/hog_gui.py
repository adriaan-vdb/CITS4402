
from PyQt6.uic import loadUi
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QPixmap

from sklearn import svm
import pickle
import os
import numpy as np


from swapHandler import swapHandler
import importlib, ablation_HOG as hogpipe 

class HOG_gui(QMainWindow):


    def __init__(self,widget):
        super(HOG_gui,self).__init__()
        loadUi("HOGConfigWindow.ui",self)
        self.swap=swapHandler(widget,self)
        filter=["sobel" , "scharr" , "prewitt" , "roberts" , "dog"]
        block=["l2hys" , "l2" , "l1" , "none"]
        colour=["gray" , "rgb" , "lab" , "ycrcb"]
        self.FilterType.addItems(filter)
        self.BlockNorm.addItems(block)
        self.ColourSpace.addItems(colour)
        self.Process.clicked.connect(self.process)
        # self.FilterType.currentTextChanged.connect(self.Ufilter)
        # self.BlockNorm.currentTextChanged.connect(self.UBNorm)
        # self.ColourSpace.currentTextChanged.connect(self.UColour)
        # self.SignedorUnsigned.stateChanged.connect(self.Usigned)
        # self.SaveDescriptor.stateChanged.connect(self.Udesc)
        # self.BinSize.valueChanged.connect(self.swapImage)
        # self.CellSize.valueChanged.connect(self.swapImage)
        # self.PHOGLevel.valueChanged.connect(self.swapImage)
        # self.GaussianSigma.valueChanged.connect(self.swapImage)
        # self.MaxImage.valueChanged.connect(self.swapImage)

        # exit_action = QAction('Exit', self)
        # exit_action.triggered.connect(self.toHOG)

        # # self.menuHOGModify.clicked.connect(self.toHOG)
        # self.Select.addAction(exit_action)
        # # self.menuModelBuild.clicked.connect(self.toBuild)
        # # self.menuevaluate.clicked.connect(self.toEval)

    def process(self):
        hogpipe.CONFIG.update(
    # ── HOG core ────────────────────────────────────────────────────────────
            GRADIENT=self.FilterType.currentText(),      # sobel | scharr | prewitt | roberts | dog
            UNSIGNED=self.SignedorUnsigned.checkState(),          # 0–180° bins (True) vs. 0–360° (False)
            BIN_SIZE=self.BinSize.value(),             # number of orientation bins per cell
            CELL_SIZE=self.CellSize.value(),            # pixels per cell
            BLOCK_NORM=self.BlockNorm.currentText(),     # l2 | l2hys | l1 | none
            PHOG_LEVELS=self.PHOGLevel.value(),          # add 2 levels of pyramid-HOG (0 = off)

            # ── Pre-processing ─────────────────────────────────────────────────────
            COLOR_SPACE=self.ColourSpace.currentText(),     # gray | rgb | lab | ycrcb
            GAMMA=self.Gamma.value(),              # 1.0 = off; <1 brightens shadows
            GAUSSIAN_SIGMA=self.GaussianSigma.value(),     # 0.0 = no blur; 0.5–1.0 typical

            # ── Dataset / I/O tweaks ───────────────────────────────────────────────
            MAX_IMAGES=self.MaxImage.value(),       # cap rows read from the CSV
            SAVE_DESCRIPTOR=self.SaveDescriptor.checkState(),   # save the 1-D feature vector too -> Need this if using FD like we are
        )
        # print("ran")
        # ④ run the extractor
        hogpipe.run_for_csv(hogpipe.CONFIG)
    # def Ufilter(self,s):
    #     hogpipe.CONFIG.update(GRADIENT=s)
    # def UBNorm(self,s):
    #     hogpipe.CONFIG.update(BLOCK_NORM=s)
    # def UColour(self,s):
    #     hogpipe.CONFIG.update(COLOR_SPACE=s)
    # def Usigned(self,s):
    #     hogpipe.CONFIG.update(UNSIGNED=s)
    # def Udesc(self,s):
    #     hogpipe.CONFIG.update(SAVE_DESCRIPTOR=s)
    # def Ubin(self,s):
    #     hogpipe.CONFIG.update(BIN_SIZE=s)
    #     print("1")
    # def Ugauss(self,s):
    #     hogpipe.CONFIG.update(BIN_SIZE=s)

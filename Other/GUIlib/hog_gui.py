from PyQt6.uic import loadUi
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QPixmap

from sklearn import svm
import pickle
import os
import numpy as np

import sys
sys.path.insert(1,"./CITS4401/Other/GUI")
from Other.GUIlib.swapHandler import swapHandler
# import importlib, ablation_HOG as hogpipe 
from Other.ablation_hog import custom_hog
from PIL import Image
from sklearn.model_selection import train_test_split




class HOG_gui(QMainWindow):


    def __init__(self,widget):
        super(HOG_gui,self).__init__()
        loadUi("Other/GUIlib/HOGConfigWindow.ui",self)
        self.swap=swapHandler(widget,self)
        # filter=["sobel" , "scharr" , "prewitt" , "roberts" , "dog"]
        block=["l2hys" , "l2" , "l1" , "none"]
        # colour=["gray" , "rgb" , "lab" , "ycrcb"]
        # self.FilterType.addItems(filter)
        self.BlockNorm.addItems(block)
        # self.ColourSpace.addItems(colour)
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

        self.nav_eval = QtWidgets.QPushButton("Go to Evaluate", self)
        self.nav_eval.setGeometry(10, 500, 120, 30)
        self.nav_eval.clicked.connect(lambda: widget.setCurrentIndex(0))

        self.nav_build = QtWidgets.QPushButton("Go to Build", self)
        self.nav_build.setGeometry(140, 500, 120, 30)
        self.nav_build.clicked.connect(lambda: widget.setCurrentIndex(1))

        self.nav_hog = QtWidgets.QPushButton("Go to HOG", self)
        self.nav_hog.setGeometry(270, 500, 120, 30)
        self.nav_hog.clicked.connect(lambda: widget.setCurrentIndex(2))

    def process(self):
        human_dir = "Analysis/data/processed/human"
        nonhuman_dir = "Analysis/data/processed/nonhuman"

        features = []
        labels = []
        MAX_IMAGES=self.MaxImage.value()/2
        num=0
        # Process human images (label = 1)
        for fname in os.listdir(human_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                num=num+1
                path = os.path.join(human_dir, fname)
                features.append((self.extract_hog_features(path),fname))
                labels.append(1)
                # np.savetxt("people/"+fname+".txt", np.array(hog_image), delimiter=',')
            if num>=MAX_IMAGES:
                break
            

        num=0
        # Process nonhuman images (label = 0)
        for fname in os.listdir(nonhuman_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                num=num+1
                path = os.path.join(nonhuman_dir, fname)
                features.append((self.extract_hog_features(path),fname))
                labels.append(0)
            if num>=MAX_IMAGES:
                break


        vals=list(range(len(labels)))
        # print(vals)
        # print(labels)
        features_train, features_test, labels_train, labels_test = train_test_split(
            vals, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
        )   
        for i in range(len(labels_train)):
            # print
            np.savetxt("GUI/Output/TrainsetGUI/"+features[features_train[i]][1]+str(labels_train[i])+".txt", np.array(features[features_train[i]][0]), delimiter=',')
        for i in range(len(labels_test)):
            np.savetxt("GUI/Output/TestsetGUI/"+features[features_test[i]][1]+str(labels_test[i])+".txt", np.array(features[features_test[i]][0]), delimiter=',')
    #     hogpipe.CONFIG.update(
    # # ── HOG core ────────────────────────────────────────────────────────────
    #         GRADIENT=self.FilterType.currentText(),      # sobel | scharr | prewitt | roberts | dog
    #         UNSIGNED=self.SignedorUnsigned.checkState(),          # 0–180° bins (True) vs. 0–360° (False)
    #         BIN_SIZE=self.BinSize.value(),             # number of orientation bins per cell
    #         CELL_SIZE=self.CellSize.value(),            # pixels per cell
    #         BLOCK_NORM=self.BlockNorm.currentText(),     # l2 | l2hys | l1 | none
    #         PHOG_LEVELS=self.PHOGLevel.value(),          # add 2 levels of pyramid-HOG (0 = off)

    #         # ── Pre-processing ─────────────────────────────────────────────────────
    #         COLOR_SPACE=self.ColourSpace.currentText(),     # gray | rgb | lab | ycrcb
    #         GAMMA=self.Gamma.value(),              # 1.0 = off; <1 brightens shadows
    #         GAUSSIAN_SIGMA=self.GaussianSigma.value(),     # 0.0 = no blur; 0.5–1.0 typical

    #         # ── Dataset / I/O tweaks ───────────────────────────────────────────────
    #         MAX_IMAGES=self.MaxImage.value(),       # cap rows read from the CSV
    #         SAVE_DESCRIPTOR=self.SaveDescriptor.checkState(),   # save the 1-D feature vector too -> Need this if using FD like we are
    #     )
    #     # print("ran")
    #     # ④ run the extractor
    #     hogpipe.run_for_csv(hogpipe.CONFIG)
    # # def Ufilter(self,s):
    # #     hogpipe.CONFIG.update(GRADIENT=s)
    # # def UBNorm(self,s):
    # #     hogpipe.CONFIG.update(BLOCK_NORM=s)
    # # def UColour(self,s):
    # #     hogpipe.CONFIG.update(COLOR_SPACE=s)
    # # def Usigned(self,s):
    # #     hogpipe.CONFIG.update(UNSIGNED=s)
    # # def Udesc(self,s):
    # #     hogpipe.CONFIG.update(SAVE_DESCRIPTOR=s)
    # # def Ubin(self,s):
    # #     hogpipe.CONFIG.update(BIN_SIZE=s)
    # #     print("1")
    # # def Ugauss(self,s):
    # #     hogpipe.CONFIG.update(BIN_SIZE=s)

    def extract_hog_features(self, img_path):
        img = Image.open(img_path).convert("L").resize((64,128))
        img_np = np.array(img)
        
        # hog_features = custom_hog(img_np, orientations=self.BinSize.value(), pixels_per_cell=(self.CellSize.value(), self.CellSize.value()),
        #                         block_norm=self.BlockNorm.currentText(), transform_sqrt=True,
        #                         feature_vector=True, unsigned=self.SignedorUnsigned.checkState(),
        #                         gamma=self.Gamma.value(), gaussian_sigma=self.GaussianSigma.value())
        hog_features = custom_hog(img_np, orientations=self.BinSize.value(), pixels_per_cell=(self.CellSize.value(), self.CellSize.value()),
                                block_norm=self.BlockNorm.currentText(), transform_sqrt=self.transform_sqrt.checkState(),
                                feature_vector=True)
        # hog_features = hog(
        #     img_np,
        #     orientations=9,
        #     pixels_per_cell=(8, 8),
        #     cells_per_block=(2, 2),
        #     block_norm='L2-Hys',
        #     feature_vector=True)

        return hog_features

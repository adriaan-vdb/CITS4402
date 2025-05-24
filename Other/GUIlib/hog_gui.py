# Gui for generating training and  test sets ogf HOG's to build and test models on


#Model train imports
from sklearn import svm
from Other.ablation_hog import custom_hog
from sklearn.model_selection import train_test_split

#File modification and creation
import pickle
import os
import numpy as np
import sys
sys.path.insert(1,"./CITS4401/Other/GUIlib")

#GUI Imports
from Other.GUIlib.swapHandler import swapHandler
from PyQt6.uic import loadUi
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QPixmap
from PIL import Image





class HOG_gui(QMainWindow):


    def __init__(self,widget):
        super(HOG_gui,self).__init__()
        loadUi("Other/GUIlib/HOGConfigWindow.ui",self)
        # Inititaiise menu bar
        self.swap=swapHandler(widget,self)
        # Add values to blocknorm selector
        block=["l2hys" , "l2" , "l1" , "none"]
        self.BlockNorm.addItems(block)
        self.Process.clicked.connect(self.process)

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

    #Bound to process button
    # Generates HOG's
    def process(self):
        #Import and label Data
        human_dir = "Other/data/raw/human"
        nonhuman_dir = "Other/data/raw/nonhuman"

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

        #Split data into training and test sets
        vals=list(range(len(labels)))

        features_train, features_test, labels_train, labels_test = train_test_split(
            vals, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
        )   
        #Save HOG's into respective training and test sets 
        for i in range(len(labels_train)):

            np.savetxt("Other/GUIlib/Output/TrainsetGUI/"+features[features_train[i]][1]+str(labels_train[i])+".txt", np.array(features[features_train[i]][0]), delimiter=',')
        for i in range(len(labels_test)):
            np.savetxt("Other/GUIlib/Output/TestsetGUI/"+features[features_test[i]][1]+str(labels_test[i])+".txt", np.array(features[features_test[i]][0]), delimiter=',')

    # Generate HOG from Image
    def extract_hog_features(self, img_path):
        img = Image.open(img_path).convert("L").resize((64,128))
        img_np = np.array(img)
        
        hog_features = custom_hog(img_np, orientations=self.BinSize.value(), pixels_per_cell=(self.CellSize.value(), self.CellSize.value()),
                                block_norm=self.BlockNorm.currentText(), transform_sqrt=self.transform_sqrt.checkState(),
                                feature_vector=True)

        return hog_features

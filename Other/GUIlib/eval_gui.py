# The Main Page of the GUI used for importing models and test data to evaluate the model



#Model train imports
from sklearn import svm
import cv2
from skimage.feature import hog
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

#File modification and creation
import joblib
import os
import numpy as np
import sys
sys.path.insert(1,"./CITS4401/Other/GUIlib")

#GUI Imports
from PyQt6.uic import loadUi
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.QtGui import QPixmap
import openpyxl
from openpyxl.drawing.image import Image
from Other.GUIlib.swapHandler import swapHandler

# HOG parameters (use the same ones as in training)
hog_params = {
    "orientations": 18,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": 'L1-sqrt'
}

# Function to preprocess the image with aspect ratio preservation
def preprocess_image(image_path, target_size=(128, 64)):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to read")
    
    # Get original dimensions
    h, w = img.shape
    
    # Calculate scale factor to preserve aspect ratio
    scale_factor = max(target_size[0] / h, target_size[1] / w)
    
    # Resize the image
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Center-crop to target size (128, 64)
    start_y = (new_h - target_size[0]) // 2
    start_x = (new_w - target_size[1]) // 2
    img_cropped = img_resized[start_y:start_y + target_size[0], start_x:start_x + target_size[1]]
    
    return img_cropped

# Function for Computing The metric used ot based on true and predicted values
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    miss_rate = 1 - recall

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Miss Rate": miss_rate,
        "False Positives": fp,
        "False Positive Rate": fp / (fp + tn)
    }

# For creating the DET Curve
def evaluate_thresholds(y_true, decision_scores, thresholds):
    results = []
    for thresh in thresholds:
        y_pred_thresh = (decision_scores >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        results.append({
            'threshold': thresh,
            'FPR': fpr,
            'FNR': fnr,
        })
    return results

# Steps taken to evaluate the 
def evaluate_model(clf, X_test, y_test):

    # Predict on test set
    pred = clf.predict(X_test)

    # Compute metrics
    test_metrics = compute_metrics(y_test, pred)

    # Get decision scores on test set
    decision_scores = clf.decision_function(X_test)

    # Define thresholds to sweep — e.g., between min and max scores
    thresholds = np.linspace(decision_scores.min(), decision_scores.max(), 100)

    # Evaluate FPR and FNR at each threshold
    threshold_results = evaluate_thresholds(y_test, decision_scores, thresholds)

    return pred,test_metrics, threshold_results



class Eval_gui(QMainWindow):


    def __init__(self,widget):
        self.doeval=True
        super(Eval_gui,self).__init__()
        loadUi("Other/GUIlib/EvaluateWindow.ui",self)
        self.Model_Import.clicked.connect(self.import_M)
        self.Model_Evaluate.clicked.connect(self.evaluate_M)
        self.Select_Test_Set.setPlainText("Test_Examples")
        self.Select_Model.setPlainText("Other/svm_model.pkl")
        self.Select_Test_SetButton.clicked.connect(self.set_clicked)
        self.Select_ModelButton.clicked.connect(self.mod_clicked)
        # Inititaiise menu bar
        self.swap=swapHandler(widget,self)
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

    # Bound to import button
    # Imports model and test set based on file path
    def import_M(self):
        self.doeval=True
        self.Model_Import.setEnabled(False)
        with open(self.Select_Model.toPlainText(), 'rb') as f:
            self.model = joblib.load(f)
        
        self.testdata=[]
        self.trueval=[]
        self.path=[]
        # stores how many true and false values stores
        Tnum=0
        Fnum=0
        # Scan each item in dir
        for entry in os.scandir(self.Select_Test_Set.toPlainText()):
            print(entry.name)
            #For HOGs
            if entry.name.endswith("1.txt") and Tnum<100:
                self.testdata.append(np.loadtxt(entry.path, delimiter=',').flatten())
                Tnum=Tnum+1 
                self.trueval.append(1)
                self.path.append("Other/data/processed/human_resized/"+entry.name[:-5])
            elif entry.name.endswith("0.txt") and Fnum<100:
                self.testdata.append(np.loadtxt(entry.path, delimiter=',').flatten())
                Fnum=Fnum+1
                self.trueval.append(0)
                self.path.append("Other/data/processed/nonhuman/"+entry.name[:-5])
            # For Images
            elif entry.name=="human" or entry.name=="human_resized":
                self.processdir((self.Select_Test_Set.toPlainText()+f"/{entry.name}"),1)
            elif entry.name=="nonhuman":
                self.processdir((self.Select_Test_Set.toPlainText()+"/nonhuman"),0)
            elif entry.name.endswith((".jpg", ".png", ".jpeg")) :
                #Can't estimate accuracy anymore
                self.doeval=False
                # Preprocess the image with aspect ratio preservation
                img_padded = preprocess_image(entry.path)

                # Extract HOG features
                self.testdata.append(hog(img_padded, **hog_params))
                self.trueval.append("unknown")
                self.path.append(entry.path)

        
        self.Model_Evaluate.setEnabled(True)
        self.Model_Import.setEnabled(True)

        print("Import Successful")
        print(self.Select_Test_Set.toPlainText())
        print(self.Select_Model.toPlainText())
        print(self.path)
    
    # Bound to evaluate button
    # Evaluates imported model based on imported test set
    def evaluate_M(self):
        self.Model_Evaluate.setEnabled(False)
        print("seems good")
        if self.doeval:
            # Evaluate accuracy
            self.q,testmetrics,other=evaluate_model(self.model,self.testdata,self.trueval)
            self.Acc.setText("Accuracy: "+str(round(testmetrics["Accuracy"]*100,2))+"%")
            self.Report.setPlainText(classification_report(self.trueval, self.q))
        else:
            self.q=self.model.predict(self.testdata)
            self.Acc.setText("Accuracy: unknown")
            self.Report.setPlainText("")

        # Displaye predicted expected
        self.PRE_T.setText("Expected: "+str(self.trueval[0]))
        self.PRE_E.setText("Predicted: "+str(self.q[0]))
        # show Image
        pix=QPixmap(self.path[0])
        self.Image.setPixmap(pix)
        self.Image.show()
        #Generate Output Excel doc
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
        workbook.save('predictions.xlsx')

        self.Model_Evaluate.setEnabled(True)
        self.Image_select.setEnabled(True)
        self.Image_select.setRange(0,len(self.testdata)-1)
        self.Image_select.valueChanged.connect(self.swapImage)

    # Bound to spin down to swap between images
    def swapImage(self,value):
        self.PRE_T.setText("Expected: "+str(self.trueval[value]))
        self.PRE_E.setText("Predicted: "+str(self.q[value]))
        pix=QPixmap(self.path[value])
        self.Image.setPixmap(pix)

    #recursive call for finding Images and HOG in subdir
    def processdir(self,dir,val):
        for entry in os.scandir(dir):
                print(entry.name)
                if entry.name.endswith("1.txt") and Tnum<100:
                    self.testdata.append(np.loadtxt(entry.path, delimiter=',').flatten())
                    Tnum=Tnum+1 
                    self.trueval.append(1)
                    self.path.append("Other/data/processed/human_resized/"+entry.name[:-5])
                elif entry.name.endswith("0.txt") and Fnum<100:
                    self.testdata.append(np.loadtxt(entry.path, delimiter=',').flatten())
                    Fnum=Fnum+1
                    self.trueval.append(0)
                    self.path.append("Other/data/processed/nonhuman/"+entry.name[:-5])
                elif entry.name=="human":
                    self.processdir((self.Select_Test_Set.toPlainText()+"/human"),1)
                elif entry.name=="nonhuman":
                    self.processdir((self.Select_Test_Set.toPlainText()+"/nonhuman"),0)
                elif entry.name.endswith((".jpg", ".png", ".jpeg")):
                    # Preprocess the image with aspect ratio preservation
                    img_padded = preprocess_image(entry.path)

                    # Extract HOG features
                    self.testdata.append(hog(img_padded, **hog_params))
                    self.trueval.append(val)
                    self.path.append(entry.path)
    
    #Opens test set file dialog
    def set_clicked(self):
      file_dialog = QFileDialog(self)
      file_dialog.setWindowTitle("Select Directory")
      file_dialog.setFileMode(QFileDialog.FileMode.Directory)
      file_dialog.setViewMode(QFileDialog.ViewMode.List)
      if file_dialog.exec():
        selected_directory = file_dialog.selectedFiles()[0]
        self.Select_Test_Set.setPlainText(selected_directory)
    
    #Opens model file dialog
    def mod_clicked(self):
      file_dialog = QFileDialog(self)
      file_dialog.setWindowTitle("Select Model")
      file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
      file_dialog.setViewMode(QFileDialog.ViewMode.List)
      if file_dialog.exec():
        selected_directory = file_dialog.selectedFiles()[0]
        self.Select_Model.setPlainText(selected_directory)
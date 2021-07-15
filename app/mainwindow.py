# Modified by Augmented Startups & Geeky Bee
# October 2020
# Facial Recognition Attendence GUI
# Full Course - https://augmentedstartups.info/yolov4release
# *-
import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
# import resource
# from model import Model
from yolo_video import Ui_OutputDialog
from pathlib import Path


class Ui_Dialog(QDialog):
    def __init__(self):
        super(Ui_Dialog, self).__init__()
        loadUi("mainwindow.ui", self)

        self.runButton.clicked.connect(self.runSlot)
        self.btnChoose.clicked.connect(self.chooseVideo)
        self.btnChooseWeight.clicked.connect(self.chooseWeight)
        self.btnChooseCfg.clicked.connect(self.chooseCfg)
        self.btnChooseData.clicked.connect(self.chooseDataCfg)

        self.rdbtnCamUSb.toggled.connect(self.onClickCamUSb)
        self.rdbCamIP.toggled.connect(self.onClickCameraIP)
        self.rdbSmartWebCam.toggled.connect(self.onClickSmartWebCam)
        self.rdbVideo.toggled.connect(self.onClickVideo)
        self.rdbImages.toggled.connect(self.onClickImages)
        self.edtAddCamera.setEnabled(False)
        self.edtSmartWebcam.setEnabled(False)
        self.edtCameraUsb.setEnabled(True)
        self.edtVideo.setEnabled(False)
        self.btnChoose.setEnabled(False)

        self._new_window = None
        self.Videocapture_ = None

    def onClickCamUSb(self):
        if self.rdbtnCamUSb.isChecked():
            print("choose webcame usb")
            self.edtAddCamera.setEnabled(False)
            self.edtSmartWebcam.setEnabled(False)
            self.edtCameraUsb.setEnabled(True)
            self.edtVideo.setEnabled(False)
            self.btnChoose.setEnabled(False)

    
    def onClickCameraIP(self):
        if self.rdbCamIP.isChecked():
            print("choose webcame camera IP")
            self.edtAddCamera.setEnabled(True)
            self.edtSmartWebcam.setEnabled(False)
            self.edtCameraUsb.setEnabled(False)
            self.edtVideo.setEnabled(False)
            self.btnChoose.setEnabled(False)

    def onClickSmartWebCam(self):
        if self.rdbSmartWebCam.isChecked():
            print("choose webcame camera IP")
            self.edtAddCamera.setEnabled(False)
            self.edtSmartWebcam.setEnabled(True)
            self.edtCameraUsb.setEnabled(False)
            self.edtVideo.setEnabled(False)
            self.btnChoose.setEnabled(False)
        
    def onClickVideo(self):
        if self.rdbVideo.isChecked():
            print("choose webcame camera IP")
            self.edtAddCamera.setEnabled(False)
            self.edtSmartWebcam.setEnabled(False)
            self.edtCameraUsb.setEnabled(False)
            self.edtVideo.setEnabled(False)
            self.btnChoose.setEnabled(False)

    def onClickImages(self):
        if self.rdbImages.isChecked():
            print("choose webcame camera IP")
            self.edtAddCamera.setEnabled(False)
            self.edtSmartWebcam.setEnabled(False)
            self.edtCameraUsb.setEnabled(False)
            self.edtVideo.setEnabled(True)
            self.btnChoose.setEnabled(True)

    def refreshAll(self):
        """
        Set the text of lineEdit once it's valid
        """
        self.Videocapture_ = "0"

    def chooseWeight(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getExistingDirectory()", "","Weight (*.weights);;All Files (*)", options=options)
        if fileName:
            print(fileName)
            self.edtWeight.setText(fileName)
    
    def chooseCfg(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getExistingDirectory()", "","Configuration (*.cfg);;All Files (*)", options=options)
        if fileName:
            print(fileName)
            self.edtCfg.setText(fileName)
    
    def chooseDataCfg(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getExistingDirectory()", "","Data (*.data);;All Files (*)", options=options)
        if fileName:
            print(fileName)
            self.edtDataCfg.setText(fileName)

    def chooseVideo(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getExistingDirectory()", "","video Files (*.mp4);;All Files (*)", options=options)
        if fileName:
            print(fileName)
            self.edtVideo.setText(fileName)

    @pyqtSlot()
    def runSlot(self):
        """
        Called when the user presses the Run button
        """
        print("Clicked Run")
        # self.refreshAll()
        if self.rdbtnCamUSb.isChecked():
            self.Videocapture_ = self.edtCameraUsb.text()
        if self.rdbCamIP.isChecked():
            self.Videocapture_ = self.edtAddCamera.text()  # "rtsp://admin:IUVCOV@192.168.0.110:554"
        if self.rdbSmartWebCam.isChecked():
            self.Videocapture_ = self.edtSmartWebcam.text()
        if self.rdbVideo.isChecked():
            self.Videocapture_ = self.edtVideo.text()
        if self.rdbImages.isChecked():
            self.Videocapture_ = self.edtVideo.text()    

        print(self.Videocapture_)
        ui.hide()  # hide the main window
        self.outputWindow_()  # Create and open new output window

    def outputWindow_(self):
        """
        Created new window for vidual output of the video in GUI
        """
        self._new_window = Ui_OutputDialog()
        self._new_window.show()
        # self._new_window.startVideo(self.Videocapture_)
        #self._new_window.loadConfiguration(self.Videocapture_,self.rdbImages)
        self._new_window.loadConfiguration(self.Videocapture_, self.edtWeight.text(), self.edtCfg.text(), self.edtDataCfg.text())
        print("Load configurations")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_Dialog()
    ui.show()
    sys.exit(app.exec_())

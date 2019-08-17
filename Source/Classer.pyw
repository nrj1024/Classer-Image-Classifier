import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QSystemTrayIcon, QFileDialog
from PyQt5.QtGui import QCursor, QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer
import ctypes
from classifier import CNN
from time import sleep
from os import listdir
from threading import Thread

class classer:
   def __init__(self):
      self.w=QWidget()
      self.w.setFixedSize(640,480)
      self.w.setWindowTitle("Classer")
      self.l=QLabel(self.w)
      self.l.setGeometry(0,462,640,18)
      self.l.setText('  Ready!')
      self.model=CNN()
      self.folder=''

   def trainM(self):
      self.datadir=''
      self.datadir = str(QFileDialog.getExistingDirectory(self.w,"Select Directory for Dataset"))
      self.folder = self.datadir.split('/')[-1]
      self.sub = listdir(self.datadir)
      if self.datadir!='':
         if 'train' not in self.sub and 'validation' not in self.sub:
            self.l.setText('  Selected directory doesn\'t have \'train\' or \'validation\' or both!')
         else:
            self.l.setText('  Training...')
            self.model.setpath(self.datadir)
            self.model.prepdata()
            self.model.initmodel()
            self.model.trainmodel()
            self.l.setText('  Model Trained for '+self.folder)

   def loadM(self):
      self.datadir=''
      self.datadir = str(QFileDialog.getExistingDirectory(self.w,"Select Directory for Test Images"))

      def load():
         self.model.setpath(self.datadir)
         self.model.prepdata()
         self.model.initmodel()
         self.model.loadmodel()

      if self.datadir!='':
         self.sub = listdir(self.datadir)
         if 'model.h5' not in self.sub:
            self.l.setText('  Model not found!')
         else:
            self.l.setText('  Loading...')
            load()
            self.folder = self.datadir.split('/')[-1]
            self.l.setText('  Model Loaded for '+self.folder)

   def mtest(self):
      if self.folder=='':
         self.l.setText('  No model is loaded yet!')
      else:
         self.testdir=''
         self.testdir = str(QFileDialog.getExistingDirectory(self.test,"Select Directory for Dataset"))
         if self.testdir!='':
            self.filelist=listdir(self.testdir)
            if self.filelist==[]:
               self.l.setText('  No images in the selected folder')      
            else:
               self.model.masstest(self.testdir)

   def stest(self):
      if self.folder=='':
         self.l.setText('  No model is loaded yet!')
      else:
         self.imgloc = QFileDialog.getOpenFileName(self.test,"Select an Image",filter="Image Files (*.jpg *.png *.bmp)")
         if self.imgloc[0]!='':
            self.model.singletest(self.imgloc[0])

   def testM(self):
      self.test=QWidget()
      self.test.move(355,75)
      self.test.setFixedSize(640,480)
      self.test.setWindowTitle("Test Model")

      self.masstest=QPushButton(self.test)
      self.masstest.setText('Mass Test')
      self.masstest.move(225,195)
      self.masstest.setCursor(QCursor(Qt.PointingHandCursor))
      self.masstest.clicked.connect(self.mtest)

      self.singletest=QPushButton(self.test)
      self.singletest.setText('Single Test')
      self.singletest.move(225,255)
      self.singletest.setCursor(QCursor(Qt.PointingHandCursor))
      self.singletest.clicked.connect(self.stest)

      self.test.show()
      


   def splashScreen(self):
      img = QLabel(self.w)
      img.setGeometry(0,0,640,480)
      pixmap = QPixmap('SplashScreen.png')
      img.setPixmap(pixmap.scaled(640,480,Qt.KeepAspectRatio))
      QTimer.singleShot(4000, img.hide)

   def mainScreen(self):
      train=QPushButton(self.w)
      train.setText('Train Model')
      train.move(225,165)
      train.setCursor(QCursor(Qt.PointingHandCursor))
      train.clicked.connect(self.trainM)

      load=QPushButton(self.w)
      load.setText('Load Model')
      load.move(225,225)
      load.setCursor(QCursor(Qt.PointingHandCursor))
      load.clicked.connect(self.loadM)

      test=QPushButton(self.w)
      test.setText('Test Model')
      test.move(225,285)
      test.setCursor(QCursor(Qt.PointingHandCursor))
      test.clicked.connect(self.testM)
   def run(self):
      self.mainScreen()
      self.splashScreen()
      self.w.show()
      sys.exit(app.exec_())
	
if __name__ == '__main__':
   myappid = 'neeraj.classer.imageclassifier'
   ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
   app = QApplication([])
   app.setStyleSheet(open('StyleSheet.css').read())
   app.setWindowIcon(QIcon('icon.png'))
   instance=classer()
   instance.run()
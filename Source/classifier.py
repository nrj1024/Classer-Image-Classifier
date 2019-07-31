import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import numpy as np
from skimage import transform
import os
from math import ceil
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class CNN:

    def __init__(self):
        self.IMAGE_SIZE = 200
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = self.IMAGE_SIZE, self.IMAGE_SIZE
        self.EPOCHS = 100
        self.BATCH_SIZE = 32
        self.TEST_SIZE = 1

        self.input_shape = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)

        self.training_data_generator = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True)
        self.validation_data_generator = ImageDataGenerator(rescale=1./255)
        self.test_data_generator = ImageDataGenerator(rescale=1./255)

    def scrollPlot(self,fig):
        self.widget = QtWidgets.QWidget()
        self.widget.resize(800,600)
        self.widget.setWindowTitle("Mass Test Results")
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)
        self.widget.layout().addWidget(self.scroll)

        self.widget.show()
    
    def setpath(self,location):
        self.directory=location
        self.training_data_dir = self.directory+'/train'
        self.validation_data_dir = self.directory+'/validation'
        self.test_data_dir = self.directory+'/test'
        self.MODEL_FILE=self.directory+'/model.h5'

    def prepdata(self):
        self.training_generator = self.training_data_generator.flow_from_directory(
            self.training_data_dir,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode="sparse")
        self.validation_generator = self.validation_data_generator.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode="sparse")
        self.classes=dict([(value, key) for key, value in self.training_generator.class_indices.items()])

    def initmodel(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(len(self.classes), activation='softmax'))
            
        self.model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='nadam',
                    metrics=['accuracy'])
                
    def trainmodel(self):
        #self.earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        #self.mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        
        if len(self.validation_generator.filenames)<self.BATCH_SIZE:
            self.BATCH_SIZE=len(self.validation_generator.filenames)

        self.model.fit_generator(
            self.training_generator,
            steps_per_epoch=len(self.training_generator.filenames) // self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=self.validation_generator,
            validation_steps=len(self.validation_generator.filenames) // self.BATCH_SIZE,
            #callbacks=[self.earlyStopping, self.mcp_save], 
            verbose=1)
        self.model.save_weights(self.MODEL_FILE)

    def loadmodel(self):
        self.model.load_weights(self.MODEL_FILE)

    def processimg(self,filename):
            self.img = Image.open(filename)
            self.img = np.array(self.img).astype('float32')/255
            self.img = transform.resize(self.img, (200, 200, 3))
            self.img = np.expand_dims(self.img, axis=0)
            return self.img

    def show_figure(self,fig):  
        dummy = plt.figure(figsize=(4,5))
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = fig
        fig.set_canvas(new_manager.canvas)

    def singletest(self,path):
        self.image = self.processimg(path)
        fig1=plt.figure(figsize=(4,5))
        plt.imshow(mpimg.imread(path))
        self.probability=self.model.predict(self.image)
        self.probability=self.probability.flatten()
        self.probability=self.probability.tolist()
        self.imgtitle='\n'
        for i in self.probability:
            if i==max(self.probability):
                self.imgtitle+='[ '+self.classes[self.probability.index(i)]+': '+"%.2f"%(i*100)+"% ]\n"
            else:
                self.imgtitle+='  '+self.classes[self.probability.index(i)]+': '+"%.2f"%(i*100)+"%\n"
        print(self.imgtitle)
        plt.axis('off')
        plt.title(self.imgtitle)
        self.show_figure(fig1)
        fig1.show()

    def masstest(self,path):
        self.images=os.listdir(path)
        fig2=plt.figure(figsize=(14, 8))
        self.length=len(self.images)
        columns = 5
        rows = ceil(self.length/columns)
        for i in range(0,self.length):
            self.loc=path+'/'+self.images[i]
            ax=fig2.add_subplot(rows, columns, i+1)
            ax.axis('off')
            ax.imshow(mpimg.imread(self.loc))
            img = self.processimg(self.loc)
            self.probability=self.model.predict(img)
            self.probability=self.probability.flatten()
            self.probability=self.probability.tolist()
            m=max(self.probability)
            plt.title(self.classes[self.probability.index(m)]+': '+"%.4f"%(m*100)+"%")

        self.scrollPlot(fig2)

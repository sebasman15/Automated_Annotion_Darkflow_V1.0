import cv2
from darkflow.net.build import TFNet
import numpy as np

class Object_Detector():
    """docstring for ClassName"""
    def __init__(self):
        self.options = 0
        self.tfnet = 0
        self.results = 0
        self.img = 0 
        self.file_name = ""

    #@Brief: Loads tensorflow deeplearning model
    def Set_TF(self, meta, pb_load, labels, threshold, gpu):
            options = { 'metaLoad': meta,
                        'pbLoad': pb_load, 
                        'labels': labels,
                        'threshold': threshold,
                        'gpu': gpu
         
                    }
            self.tfnet = TFNet(options)

    #@Brief: Loads the image that was last added to given folder, or loads a single image to the class for testing if the param pathname is given a path to a direct image
    #@Params[in]: Pathname to a directory or folder, show loaded image after loading.
    def Load_Latest_Img(self,pathname = "", show =True):
        print("pathname of image to load = " + pathname)        
        self.img = cv2.imread(pathname)
        cv2.imshow("loaded img", self.img)
        cv2.waitKey(10)

    """@Brief: uses trained tensorflow model to detect mushrooms, WARNING: ONLY WORKS FOR RESOLUTION TRAINING IMAGES WHERE SUPPLIED FOR"""
    def Detect(self):
        self.results = []
        self.results = self.tfnet.return_predict(self.img)
        

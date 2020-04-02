# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
from PreProcess import * 
from DbFirebase import *
class Detection:
    def __init__(self, MODEL_NAME = 'conModel',GRAPH_NAME = 'detect.tflite',LABELMAP_NAME = 'labelmap.txt',min_conf_threshold = float(0.5), IM_NAME =None,IM_DIR =None ):
        self.MODEL_NAME = MODEL_NAME
        self.GRAPH_NAME = GRAPH_NAME
        self.LABELMAP_NAME = LABELMAP_NAME
        self.min_conf_threshold = min_conf_threshold
        self.IM_NAME = IM_NAME
        self.IM_DIR = IM_DIR
    # Parse input image name and directory. 
    def analyzing(self):
        
        CWD_PATH = os.getcwd()
        output_path = os.path.join(CWD_PATH,'analyze')
        preProcess = PreProcess()
        index = preProcess.nameImage('analyze')

        # If both an image AND a folder are specified, throw an error
        if (self.IM_NAME and self.IM_DIR):
            print('you can only use IM_NAME OR IM_DIR')
            sys.exit()

        # If neither an image or a folder are specified, default to using 'test1.jpg' for image name
        if (not self.IM_NAME and not self.IM_DIR):
            self.IM_DIR = 'new'

        # Import TensorFlow libraries
        # If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tensorflow')
        if pkg is None:
            from tflite_runtime.interpreter import Interpreter
        else:
            from tensorflow.lite.python.interpreter import Interpreter

        # Get path to current working directory
        

        # Define path to images and grab all image filenames
        if self.IM_DIR:
            PATH_TO_IMAGES = os.path.join(CWD_PATH,self.IM_DIR)
            images = glob.glob(PATH_TO_IMAGES + '/*')

        elif self.IM_NAME:
            PATH_TO_IMAGES = os.path.join(CWD_PATH,self.IM_NAME)
            images = glob.glob(PATH_TO_IMAGES)

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,self.MODEL_NAME,self.GRAPH_NAME)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,self.MODEL_NAME,self.LABELMAP_NAME)

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

        # Load the Tensorflow Lite model.
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        print(width,height)

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        # Loop over every image and perform detection
        for image_path in images:
            leaf = flower = melon = 0  
            # Load image and resize to expected shape [1xHxWx3]
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape 
            image_resized = cv2.resize(image_rgb, (width, height),interpolation = cv2.INTER_AREA)
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
            #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    if(object_name =='leaf'):
                        leaf = leaf + 1
                    elif(object_name =='flower'):
                        flower = flower + 1
                    else:
                        melon = melon + 1
            # All the results have been drawn on the image, now display the image
            print('image', index, ':')
            print('leaf:', leaf)
            print('flower:', flower)
            print('melon:', melon)
            uploadToFirebase = DbFirebase(leaves=leaf, flowers=flower, melons=melon)
            uploadToFirebase.add()
            cv2.imshow('Object detector', image)
            out = os.path.join(output_path, str(index)+".jpg")
            cv2.imwrite(out, image)
            
            index = index + 1
            # Press any key to continue to next image, or press 'q' to quit
            cv2.waitKey(1) 
        preProcess.moveImage()
        # Clean up
        cv2.destroyAllWindows()
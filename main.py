import cv2, os
from time import sleep
from PreProcess import *
import argparse
from datetime import datetime
from Detection import *
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--morning', help='Folder the .tflite file is located in',
                    default='17:47:0')
parser.add_argument('--afternoon', help='Name of the .tflite file, if different than detect.tflite',
                    default='11:38:0')
parser.add_argument('--evening', help='Name of the labelmap file, if different than labelmap.txt',
                    default='11:39:0')

args = parser.parse_args()
set_time = [args.morning, args.afternoon, args.evening]
name = ''

while True:
    files= os.listdir('input')
    preProcess = PreProcess()
    index_new = preProcess.nameImage()
    d = datetime.now()
    cur_time = str(d.hour) +':'+ str(d.minute) + ':'+ str(d.second)  
    #capture
    for i in range(len(set_time)):
        if cur_time == set_time[i]:
            preProcess.captureImage()
            
    #resize & name
    if len(files)>0:
        for file in files:
            sleep(3)
            test = PreProcess(file=file, name=index_new)
            test.preImages()
            index_new = index_new +1
    if index_new > 0:
        detect = Detection()
        detect.analyzing()

   
        
        
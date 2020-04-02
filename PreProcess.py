import cv2, os, shutil
from PIL import Image
from time import sleep

class PreProcess:
    def __init__(self,infile = os.path.join(os.getcwd(), 'input'), output= os.path.join(os.getcwd(),'new'), file = os.listdir('input')[0] if (len(os.listdir('input'))>0) else 'empty', name =len(os.listdir('new'))):
        self.input = infile
        self.output = output
        self.file = file
        self.name = name
        
    def preImages(self):
        # raed image
        if self.file =='empty':
            print('no image in dir')
        else:
            img = cv2.imread(os.path.join(self.input,self.file), cv2.IMREAD_UNCHANGED)
            # define image size
            width = int(1024)
            height = int(768)
            dim = (width, height)
            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            # rename image and move image
            cv2.imwrite(os.path.join(self.output,str(self.name)+".jpg"), resized)
            # remove image
            if self.file.endswith('.jpg'):
                os.remove(os.path.join(self.input, self.file))
            # delay for 1/2 second 
            cv2.waitKey(500)
            cv2.destroyAllWindows()  

    def captureImage(self):
        index = 0
        cap = cv2.VideoCapture(1)
        _, frame = cap.read()
        output = os.path.join(self.input,str(index)+".jpg")
        cv2.imwrite(output, frame)
        cap.release()
        cv2.destroyAllWindows()
        
   
    def nameImage(self, src = 'new'):
        return len(os.listdir(src))
        
    def moveImage(self, src = 'new', dst = 'dataset'):
        input_src = self.output
        images = os.listdir(src)
        index  = len(os.listdir(dst))

        for image in images:
            input_path = os.path.join(input_src,image)
            output_path = os.path.join(dst, str(index))
            shutil.move(input_path, os.path.join(os.getcwd(), output_path)+'.jpg')
            index = index + 1
            


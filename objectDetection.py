#objectDetection.py
from imageai.Detection import ObjectDetection
from PIL import Image
import os
import sys

#Disable TF warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system('cls' if os.name == 'nt' else 'clear')
execution_path = os.getcwd()

print("Enter Image name with ext (i.e. hello.jpg):")
img = input()
#List of all possible models (as per Oct 24 2018)
modelsDir = "models/imgPrediction/"
models = ["resnet50_coco_best_v2.0.1.h5","yolo.h5","yolo-tiny.h5"]

#Variable to select the appropriate model and model type
useModel = 1

#Min probability to detect
minProb = 90

#Checking for user model override
try:
    userModel = int(sys.argv[1])
    if ( userModel < 3 and userModel > -1 ):
        useModel = userModel
except:
    pass

#Printing to console the model we are going to use
print("Detecting using {}   ...".format(models[useModel]))

#Initializing detector
detector = ObjectDetection()

#Selecting model type
if useModel == 0:
    detector.setModelTypeAsRetinaNet()
elif useModel == 1:
    detector.setModelTypeAsYOLOv3()
else:    
    detector.setModelTypeAsTinyYOLOv3()
    
detector.setModelPath( os.path.join(execution_path , modelsDir + models[useModel]))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , img), output_image_path=os.path.join(execution_path , img + "-oDetection.jpg"), minimum_percentage_probability=minProb)


for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"])
    #if you want to print box points, use below instead
    #print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    #print("--------------------------------")
    
    
#Showing original and resulting image
iImg = Image.open(img)
oImg = Image.open(img + "-oDetection.jpg")
iImg.show()
oImg.show()

#J05
#Based on FirstObjectDetection.py
#https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection#firstdetection

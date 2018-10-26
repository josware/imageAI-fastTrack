#objectDetection.py
from imageai.Detection import ObjectDetection
from PIL import Image
import os
import sys

#Disable TF warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system('cls' if os.name == 'nt' else 'clear')
execution_path = os.getcwd()





modelsDir = "models/objDetection/"
imagesDir = "pictures/"
models = ["resnet50_coco_best_v2.0.1.h5","yolo.h5","yolo-tiny.h5"]


#Checking for user override
try:
    img = imagesDir + sys.argv[1]

except:
    print("--- \nYou can run: python objectDetection.py image [prob (1-100)] [model (0-2)] ")
    print("      i.e. python objectDetection.py a.jpg 10 2 \n---\n\n")
    print("Enter image name to predict including extension (i.e. hello.jpg):")
    img = imagesDir + input()

try:
    prob= int(sys.argv[2])
    if ( prob < 101 and prob > 0 ):
        minProb = prob
    else:
        minProb = 30
except:
    #Min probability to detect
    minProb = 30
    
    

try:
    userModel = int(sys.argv[3])
    if ( userModel < 3 and userModel > -1 ):
        useModel = userModel
    else:
        useModel = 2
except:
    #Variable to select the appropriate model and model type
    useModel = 2



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

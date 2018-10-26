#imagePrediction.py
from imageai.Prediction import ImagePrediction
import os
import sys

#Disable TF warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system('cls' if os.name == 'nt' else 'clear')
execution_path = os.getcwd()


modelsDir = "models/imgPrediction/"
imagesDir = "pictures/"
#List of all possible models (as per Oct 24 2018)
models = ["squeezenet_weights_tf_dim_ordering_tf_kernels.h5","resnet50_weights_tf_dim_ordering_tf_kernels.h5",
"inception_v3_weights_tf_dim_ordering_tf_kernels.h5","DenseNet-BC-121-32.h5"]

try:
    img = imagesDir + sys.argv[1]

except:
    print("--- \nYou can also run: python imagePrediction.py image [results (1-0)] [model (0-3)] ")
    print("      i.e. python imagePrediction.py a.jpg 5 0 \n---\n\n")
    print("Enter image name to predict including extension (i.e. hello.jpg):")
    img = imagesDir + input()

try:
    res= int(sys.argv[2])
    if ( res < 10 and res > 1 ):
        resCount = res
    else:
        resCount = 5
except:
    resCount = 5
    

try:
    model = int(sys.argv[3])
    if ( model < 4 and model > -1 ):
        userModel = model
    else:
        userModel = 0
except:
    #Variable to select the appropriate model and model type
    userModel = 0
    


    
#Printing to console the model we are going to use
print("Detecting using {}   ...".format(models[userModel]))

prediction = ImagePrediction()

if userModel == 0:
    prediction.setModelTypeAsSqueezeNet()
elif userModel == 1:
    prediction.setModelTypeAsResNet()
elif userModel == 2:
    prediction.setModelTypeAsInceptionV3()
else:
    prediction.setModelTypeAsDenseNet()


prediction.setModelPath(os.path.join(execution_path, modelsDir + models[userModel]))
prediction.loadModel()


predictions, probabilities = prediction.predictImage(os.path.join(execution_path, img), result_count=resCount )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)

#J05
#Based on FirstPrediction.py
#https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Prediction#firstprediction

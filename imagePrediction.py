from imageai.Prediction import ImagePrediction
import os
import sys

#Disable TF warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system('cls' if os.name == 'nt' else 'clear')
execution_path = os.getcwd()

print("Enter Image name with ext (i.e. hello.jpg):")
img = input()
#List of all possible models (as per Oct 24 2018)
modelsDir = "models/objDetection/"
models = ["squeezenet_weights_tf_dim_ordering_tf_kernels.h5","resnet50_weights_tf_dim_ordering_tf_kernels.h5",
"inception_v3_weights_tf_dim_ordering_tf_kernels.h5","DenseNet-BC-121-32.h5"]

#Variable to select the appropriate model and model type
useModel = 1

#Min probability to detect
resCount= 5

try:
    userModel = int(sys.argv[1])
    if ( userModel < 4 and userModel > -1 ):
        useModel = userModel
except:
    pass
    
#Printing to console the model we are going to use
print("Detecting using {}   ...".format(models[useModel]))

prediction = ImagePrediction()

if useModel == 0:
    prediction.setModelTypeAsSqueezeNet()
elif useModel == 1:
    prediction.setModelTypeAsResNet()
elif useModel == 2:
    prediction.setModelTypeAsInceptionV3()
else:
    prediction.setModelTypeAsDenseNet()


prediction.setModelPath(os.path.join(execution_path, modelsDir + models[useModel]))
prediction.loadModel()


predictions, probabilities = prediction.predictImage(os.path.join(execution_path, img), result_count=resCount )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)

#J05
#Based on FirstPrediction.py
#https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Prediction#firstprediction
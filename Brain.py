from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()  # to grab the three images

# now we want to do predictions
prediction = ImageClassification()

#  we use the prediction variabile with differente models
prediction.setModelTypeAsMobileNetV2()

# we grab the model we chose
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5"))
prediction.loadModel()  # to load and use the model we chose

# we make some predictions; probabilities means how confident the model is in its prediction
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "giraffe.jpg"), result_count=5)
for each_prediction, each_probability in zip(predictions, probabilities):
    print(each_prediction, " : ", each_probability)

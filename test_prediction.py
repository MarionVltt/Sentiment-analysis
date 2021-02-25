# usr/bin/python3

# Prediction test file, uses SingleRaw and SinglePredict from preprocessing.py and prediction.py

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import preprocessing
import prediction

# Define sentence
sentence = "Good morning, I'm 21 and my web <br /><br /> .site is www.fakewebsite.com or https://www.fakewebsite.com!!"

# Initialization and preprocessing
prediction.SinglePredict.initialize(model_path="model")
new_predict = prediction.SinglePredict(
    sentence, ["Neg", "Pos"], sequence_to_remove=["book", "is"])
new_predict.preprocess()
print("Preprocesed text : ", new_predict.clean_text)

# Prediction
if not new_predict.default_case:
    new_predict.predict()
    result = new_predict.predicted_class
    print("Predicted class : ", result)
else:
    results = None
    print("default case : no prediction")

# Try a second sentence
new_predict = prediction.SinglePredict(
    "The film was great", ["Negative", "Positive"])
new_predict.preprocess()
if not new_predict.default_case:
    new_predict.predict()
    new_result = new_predict.predicted_class
    print("Predicted class : ", new_result)
else:
    new_result = None
    print("default case : no prediction")

# Try a third sentence, with the global function clean_and_predict
new_predict = prediction.SinglePredict(
    "It is great here, I love it", ["Neg", "Pos"])
new_predict.clean_and_predict()

prediction.SinglePredict.unload()

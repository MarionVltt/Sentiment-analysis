import preprocessing_en
import preprocessing_fr
import prediction_en
import prediction_fr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import time
import json

t = time.time()
prediction_fr.SinglePredict.initialize(model_path="model_dir_fr")
t0 = time.time()
new_predict = prediction_fr.SinglePredict("Ce film était ennuyeux", ["Neg", "Pos"])
t1 = time.time()
new_predict.preprocess()
print(new_predict.clean_text)
t2 = time.time()
t3 = time.time()
if not new_predict.default_case:
    t4 = time.time()
    times1 = new_predict.predict()
    results1 = new_predict.predicted_class
    t5 = time.time()
else:
    t4 = time.time()
    t5 = time.time()
    results1 = None
    print("default case : no prediction")
new_predict = prediction_fr.SinglePredict("J'aime bien ce film", ["Negative", "Positive"])
t6 = time.time()
new_predict.preprocess()
t7 = time.time()
if not new_predict.default_case:
    t8 = time.time()
    times2 = new_predict.predict()
    results2 = new_predict.predicted_class
    t9 = time.time()
else:
    t8 = time.time()
    t9 = time.time()
    results2 = None
    print("default case : no prediction")
new_predict = prediction_fr.SinglePredict("C'est agréable", ["Neg", "Pos"])
new_predict.clean_and_predict()
prediction_fr.SinglePredict.unload()
print("Time init : ", t0-t)
print("Total time needed : ", t5-t0)
print("Time creation 1: ", t1-t0)
print("Time clean : ", t2-t1)
print("Time to features : ", t4-t3)
print("Time predict : ", t5-t4)
print("\n")
print("Total time needed : ", t9-t5)
print("Time creation 2: ", t6-t5)
print("Time clean : ", t7-t6)
print("Time features : ", t8-t7)
print("Time predict : ", t9-t8)
print(results1, results2)

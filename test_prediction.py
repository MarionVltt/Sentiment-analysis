import preprocessing_en
import preprocessing_fr
import prediction_en
import prediction_fr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import time
import json

###
raw = "Good morning, I'm 21 and my web <br /><br /> .site is www.fakewebsite.com or https://www.fakewebsite.com!!"
new_raw = preprocessing_en.SingleRaw(raw)
new_raw.standardize_raw()
print(new_raw)
# file_path = "books-fr.test.csv"
# new_raw = preprocessing_fr.RangeRaw(file_path, 0)
# new_raw.standardize_range()
# print(new_raw.df.shape)
# print(new_raw.df.head())
#
# file_path = "dataAmazonElecSmall.csv"
# new_raw = preprocessing_en.RangeRaw(file_path, 1)
# new_raw.standardize_range()
# print(new_raw.df.shape)
# print(new_raw.df.head())
####
t = time.time()
prediction_en.SinglePredict.initialize(model_path="model")
t0 = time.time()
new_predict = prediction_en.SinglePredict("wwww.absurde.com", ["Neg", "Pos"], sequence_to_remove=["book", "is"])
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
#print("First done")
#new_predict = prediction_en.SinglePredict("I'm all for saving money on a case but this case was absolutely cheap and flimsy.  Broke within a week and just all around not a great buy.  The silicone doesn't stay in place either.", ["Neg", "Pos"],model_path="model_dir")
t5bis = time.time()
new_predict = prediction_en.SinglePredict("The film was great", ["Negative", "Positive"])
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
new_predict = prediction_en.SinglePredict("It is great here, I love it", ["Neg", "Pos"])
new_predict.clean_and_predict()
prediction_en.SinglePredict.unload()
print("Time init : ", t0-t)
print("Total time needed : ", t5-t0)
print("Time creation 1: ", t1-t0)
print("Time clean : ", t2-t1)
print("Time to features : ", t4-t3)
print("Time predict : ", t5-t4)
print("\n")
print("Total time needed : ", t9-t5bis)
print("Time creation 2: ", t6-t5bis)
print("Time clean : ", t7-t6)
print("Time features : ", t8-t7)
print("Time predict : ", t9-t8)
print(results1, results2)


print(new_predict.input_features[0].input_ids)
print(new_predict.input_features[0].input_mask)
print(new_predict.input_features[0].label_id)
print(new_predict.input_features[0].segment_ids)


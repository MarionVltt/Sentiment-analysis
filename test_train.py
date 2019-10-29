import prediction_en
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import time
import pandas as pd
import numpy as np
import os

print(os.getcwd())
df = pd.read_csv("data/data_en.csv", sep=";", header=None, index_col=0)
df.columns = ['Tag', 'Phrase']
df = df.reindex(np.random.permutation(df.index))
print(df.shape)
print(df.head())
df_small = df.sample(n=10)
print(df_small.Tag.value_counts())
print("Preprocesss and training")
rp = prediction_en.RangePredict(df_small, 1, 0)
rp.preprocess()
rp.train()
rp.export()
# df_test = pd.read_csv("dataAmazonElecSmall2.csv", sep=";", header=None, index_col=0)
# print(df_test.head())
# #df_test.drop(['clean_text'], inplace=True, axis=1)
# df_test.columns = ['Tag', 'Phrase']
# df_test = df_test.reindex(np.random.permutation(df_test.index))
# df_test.dropna(inplace=True)
# #df_test = df_test.sample(n=10000)
# print(df_test.shape)
# print(df_test.head())
# print("Evaluating generalization")
# results = rp.predict_estimator_test(df_test['Phrase'], df_test['Tag'])

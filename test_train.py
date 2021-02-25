# usr/bin/python3

import os

import pandas as pd
import numpy as np

import prediction


# Load data
df = pd.read_csv("data/data_en.csv", sep=";", header=None, index_col=0)
df.columns = ['Tag', 'Phrase']
df = df.reindex(np.random.permutation(df.index))
print(df.shape)
print(df.head())
df_small = df.sample(n=10)
print(df_small.Tag.value_counts())

# Training
print("Preprocesss and training")
rp = prediction_en.RangePredict(df_small, 1, 0)
rp.preprocess()
rp.train()
rp.export()

# Sentiment analysis

This module performs sentiment analysis of a sentence, 
in the sense where it tells if the sentence is rather 
positive or negative. Two trained models are provided, one for english
and one for french, along with 2 files for preprocessing (preprocessing_en and 
preprocessing_fr), and to files to train and predict (prediction_en and prediction_fr)
in one of the two language. 

It has been implemented and tested using python 3.7 and tensorflow 1.13. 

## Installation

You need to have the following libraries installed : numpy, pandas,
json, sklearn, tensorflow, tensorflow-hub and bert 

To install bert:
```bash
pip install bert-tensorflow
```

## How to make predictions

This uses a model exported from a tensorflow estimator as a .pb
file and two folders _variables_ and _assets_ (not provided here, see the train and 
export part for more details on how to create a new one).

First import the file:
```python
import prediction_en # or prediction_fr
```

The class prediction_en.SinglePredict allows you to make prediction for a
single sentence. Before using it, you have to call the static
method initialize(), which will import the model and run it 
with a first sentence (the first prediction takes longer). 
The model is then marked as loaded and can be used with several
instances of prediction_en.SinglePredict, without reloading it each time.
You can specify two arguments:
- model_path (string, default = 'model_dir'): path to the
**directory** containing the model to be used.

```python
prediction_en.SinglePredict.initialize()
```

Now you can create an instance of SinglePredict.
It takes the following arguments:
- sentence (string)
- label_list (list, default = [0, 1]): list of possible classes for the sentence,
usually [0, 1] or ['Negative', 'Positive'].
- sequence_to_remove : list of regex you want to remove from the sentences 
- max_seq_length (int, default = 128): maximum number of tokens used for the 
prediction. If the sentence is longer, it's truncated.

Only the sentence argument is mandatory, the rest is optional. 

```python
sentence = "It is so beautiful here" 
new_predict = prediction_en.SinglePredict(sentence, ["Neg", "Pos"])
```
To make prediction you can use one of the two codes below. The first one separates
the processus in two methods calls, when the second one regroups them
in one method, which will also print out the result.
This code below preprocesses the sentence, transforms it in features
understandable to bert, and makes the prediction.
During preprocessing, the text is lowered, the mention with "@" and the urls in "www." and "https://" are removed, 
as well as the sequence passed as argument of the initializer. Furthermore, all non letters characters are removed.
If the sentence is empty as the end of preprocessing, the attribute _default_case_ is
set to True so you can test it to make no prediction in that case and maybe return 
something different. 
If you are using the second technique here, a test is being made at the beginning of
predict to avoid predicting for an empty string, so 
new_predict.predicted_class stays to None. 
```python
new_predict.preprocess()
if not new_predict.default_case:
    new_predict.predict()
    print(new_predict.predicted_class)
    # Pos
else:
    print("No prediction")


# OR : 
new_predict.clean_and_predict()
# The predicted class is: Pos
```

To make a new prediction, you have to create a new instance of
SinglePredict and rerun the code above.

When you're done, execute the code below to close the tensorflow session.

```python
prediction_en.SinglePredict.unload()
```

To predict the polarity of a french sentence, you just have to change the file you're
using from prediction_en to prediction_fr.
The files test_prediction and test_prediction_fr implement the code above long with timestamps.


## How to train and export

If you want to train a model on a new dataset, here's how.

The prediction.py files contains two classes 
(other than SinglePredict) : ClassifierModel and RangePredict. ClasssifierModel
regroups all methods of the bert training procedure (explained in this tutorial:
https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb).
You don't have to worry about it, except if you want to change the default parameters
(in that case you obviously need to know which is used where). To pass 
modified parameters, you need to create a json with the format of default_params.json and pass the file as
parameter of RangePredict.

All you need to do is the following. 
First import you dataset as a pandas dataframe, with a column for the sentences,
and one for the labels (it can have additional columns, they will no be used
by the trainer). See more details about the parameters of read_csv here : 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv
```python
import pandas as pd
df = pd.read_csv(str(file_path), sep=';', header=None, index_col=0)
```

The data need to be randomly shuffled, as the order of the examples
have an impact on the results. 

Then, create and preprocess your data with the following code. The arguments
of the initializer of Range predict include at least the dataframe, the index of 
the column containing the sentences, and the index of the labels columns.
The others optional parameters are : 
- sequence_to_remove (lis, default = []): list of regex you want to remove from the sentences 
- seed (int, default = 42): the random seed to be used for train/test split, 
- model_path (see definition of SinglePredict above for details)
- text_processing (default = 1): 0 for raw text,1 for clean text
- test_size (float between 0 and 1, default = 0.2): fraction of test set in the global dataset
- param_file (default = 'default_params.json'): file path of the parameters json

```python
rp = prediction_en.RangePredict(df, 1, 0)
rp.preprocess()
```

You're ready to train ! Execute the following line:

```python
rp.train()
```

This will split the dataset in train/test sets, train and evaluate the model. 
In order to print the metrics, tf.logging.set_verbosity needs to be on INFO
level, it is set by default so comment the corresponding line in the code if you don't want them.

To evaluate your model on another dataset, do:

 ```python
results = rp.predict_estimator_test(df_test['Phrase'], df_test['Tag'])
```

It will print the principal metrics (accuracy, precision, recall, f1 and confusion
matrix).

If you don't use the preprocessing methods, you should at least run df.dropna(inplace=True) to avoid 
type errors.

Two datasets are provided, one for English with 120000 reviews on Amazon electronics and one for French with around 40000 reviews of books, musics and DVDs.
You can find more Amazon reviews here : http://jmcauley.ucsd.edu/data/amazon/

**Export**

To export the model:
 ```python
exprt_path = './export'
rp.export(export_path)
```

Warning : the exported model will be in a sub-folder (named for example 1534535436).
It's the path to that folder that you have to indicate if you want to
import the model with SinglePredict. 

## TO DO

This module is not perfect and can be improved in different ways :
- add a neutral class between negative and positive
- find a better way to distinguish the sentences for which it's useful to apply the classifier 
- increase performances of tensorflow (Tensorflowserving?)

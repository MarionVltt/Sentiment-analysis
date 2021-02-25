# usr/bin/python3

import json
import time

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.saved_model import tag_constants
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

import bert  # install with !pip install bert-tensorflow
from bert import run_classifier
from bert import optimization
from bert import tokenization

from preprocessing import RangeRaw, SingleRaw

DEBUG = False


class ClassifierModel():
    """
    Class that stores all steps from the model to ensure train and test
    are processed the same way
    Shall be instantiated only once

    Attributes:
        - self.params: contains all necessary parameters, loaded from default_params.json
            - MAX_SEQ_LENGTH (int) : maximum number of characters in the sequence
            - BATCH_SIZE (int)
            - LEARNING_RATE (float)
            - NUM_TRAIN_EPOCHS (int): number of training epochs
            - WARMUP_PROPORTION (float): proportion of the training steps dedicated to warmup (smaller learning rate)
            - SAVE_CHECKPOINTS_STEPS (int): indicates when to save a checkpoint model
            - SAVE_SUMMARY_STEPS (int)
            - BERT_MODEL_HUB (string): adress of the BERT model that should be used (eg. https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1)
            - OUTPUT_DIR (string): output directory
            - DO_DELETE (bool): if True and a directory named like OUTPUT_DIR exits, deletes it
        - self.text_preprocessing (bool): specify which version of the text should be used (raw or cleaned)
        - self.is_set_up (bool): indicates if the initialization is completed or not
        - self.classifier_trained (bool): indicates if the model have been trained or not
        - self.label_list (list): list of labels in the dataset (ag. [0,1])
        - self.tokenizer: BERT tokenizer
        - self.estimator: Tensorflow structure used here to train and predict

    Methods:
        - initialize: loads parameters and create the tokenizer
        - sentences_to_features: converts the sentences into features BERT uses
        - create_tokenizer_from_hub_module : BERT function to create the tokenizer
        - create_model (static)

    """
    TEXT_RAW = 0
    TEXT_CLEAN = 1

    BERT_MODEL_HUB = None

    def __init__(self):
        self.params = dict()
        self.text_preprocessing = None
        self.is_set_up = False
        self.classifier_trained = False
        self.label_list = []
        self.tokenizer = None
        self.estimator = None

    def initialize(self,
                   text_preprocessing=None, param_file='default_params.json'):
        """
        Its purpose is to:
            - load and set the parameters
            - create the tokenizer

        Parameters :
            - text_preprocessing (bool): specify if which version of the text whould be used (raw or cleaned)
            - param_file (string): path to the json file containing the values of the paramters

        """
        with open(param_file, 'r') as f:
            self.params = json.load(f)

        if self.params["DO_DELETE"]:
            try:
                tf.gfile.DeleteRecursively(self.params["OUTPUT_DIR"])
            except:
                # Doesn't matter if the directory didn't exist
                pass
        tf.gfile.MakeDirs(self.params["OUTPUT_DIR"])
        if DEBUG:
            print(
                '***** Model output directory: {} *****'.format(self.params["OUTPUT_DIR"]))

        if not text_preprocessing:
            self.text_preprocessing = ClassifierModel.TEXT_RAW
        else:
            self.text_preprocessing = ClassifierModel.TEXT_CLEAN

        self.is_set_up = True
        self.classifier_trained = False

        ClassifierModel.BERT_MODEL_HUB = self.params['BERT_MODEL_HUB']
        if DEBUG:
            print("Creating tokenizer")
        self.tokenizer = self.create_tokenizer_from_hub_module()
        if DEBUG:
            print("Initialization completed")

    def create_tokenizer_from_hub_module(self):
        """
        From BERT repository
        Get the vocab file and casing info from the Hub module.
        """
        with tf.Graph().as_default():
            bert_module = hub.Module(self.params["BERT_MODEL_HUB"])
            tokenization_info = bert_module(
                signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])
        return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def sentences_to_features(self, sentences, labels):
        """
        Transform the string into BERT features

        Parameters:
            - sentences (pd.Series)
            - labels (pd.Series)

        Return:
            - BERT input features
        """

        input_examples = [run_classifier.InputExample(guid="", text_a=s, text_b=None, label=l) for s, l in
                          zip(sentences, labels)]  # here, "" is just a dummy label
        input_features = run_classifier.convert_examples_to_features(input_examples, self.label_list,
                                                                     self.params["MAX_SEQ_LENGTH"],
                                                                     self.tokenizer)
        return input_features

    #####################################################################################
    # The four next methods are necessary for the estimator (code from BERT repository) #
    #####################################################################################

    @staticmethod
    def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):
        """
        Creates a classification model.

        """

        bert_module = hub.Module(
            ClassifierModel.BERT_MODEL_HUB,
            trainable=True)
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(
                labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(
                tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return (predicted_labels, log_probs)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = - \
                tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, predicted_labels, log_probs)

    @staticmethod
    def model_fn_builder(num_labels, learning_rate, num_train_steps,
                         num_warmup_steps):
        """
        Returns `model_fn` closure for TPUEstimator.
        model_fn_builder actually creates the model function
        using the passed parameters for num_labels, learning_rate, etc.
        """

        def model_fn(features, labels, mode, params):
            """The `model_fn` for TPUEstimator."""

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # TRAIN and EVAL
            if not is_predicting:

                (loss, predicted_labels, log_probs) = ClassifierModel.create_model(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                train_op = bert.optimization.create_optimizer(
                    loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

                # Calculate evaluation metrics.
                def metric_fn(label_ids, predicted_labels):
                    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                    f1_score = tf.contrib.metrics.f1_score(
                        label_ids,
                        predicted_labels)
                    auc = tf.metrics.auc(
                        label_ids,
                        predicted_labels)
                    recall = tf.metrics.recall(
                        label_ids,
                        predicted_labels)
                    precision = tf.metrics.precision(
                        label_ids,
                        predicted_labels)
                    true_pos = tf.metrics.true_positives(
                        label_ids,
                        predicted_labels)
                    true_neg = tf.metrics.true_negatives(
                        label_ids,
                        predicted_labels)
                    false_pos = tf.metrics.false_positives(
                        label_ids,
                        predicted_labels)
                    false_neg = tf.metrics.false_negatives(
                        label_ids,
                        predicted_labels)

                    return {
                        "eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "true_positives": true_pos,
                        "true_negatives": true_neg,
                        "false_positives": false_pos,
                        "false_negatives": false_neg
                    }

                eval_metrics = metric_fn(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      train_op=train_op)
                else:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      eval_metric_ops=eval_metrics)
            else:
                (predicted_labels, log_probs) = ClassifierModel.create_model(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                predictions = {
                    'probabilities': log_probs,
                    'labels': predicted_labels
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Return the actual model function in the closure
        return model_fn

    def run_config_builder(self):
        return tf.estimator.RunConfig(model_dir=self.params["OUTPUT_DIR"],
                                      save_summary_steps=self.params["SAVE_SUMMARY_STEPS"],
                                      save_checkpoints_steps=self.params["SAVE_CHECKPOINTS_STEPS"])

    def estimator_builder(self, model_fn, run_config):
        return tf.estimator.Estimator(model_fn=model_fn, config=run_config,
                                      params={"batch_size": self.params["BATCH_SIZE"]})

    def train(self, X, y):
        """
        Split X, y as train and test sets. Then convert them to features, classify and score.
        Shows train and test score to infer whether model is too simple or overfitted.

        Parameters:
        - X (pd.Series): text
        - y (pd.Series): labels
        """
        tf.logging.set_verbosity(
            tf.logging.INFO)  # comment if you don't want to display the information during training/evaluation

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.params["TEST_SIZE"], random_state=42, stratify=y)

        self.label_list = y.unique()

        train_features = self.sentences_to_features(X_train, y_train)
        test_features = self.sentences_to_features(X_test, y_test)
        if DEBUG:
            print("Transformation to features completed")

        num_train_steps = int(
            len(train_features) / self.params["BATCH_SIZE"] * self.params["NUM_TRAIN_EPOCHS"])
        num_warmup_steps = int(
            num_train_steps * self.params["WARMUP_PROPORTION"])

        run_config = self.run_config_builder()
        model_fn = self.model_fn_builder(len(self.label_list), self.params["LEARNING_RATE"], num_train_steps,
                                         num_warmup_steps)
        self.estimator = self.estimator_builder(model_fn, run_config)

        train_input_fn = bert.run_classifier.input_fn_builder(features=train_features,
                                                              seq_length=self.params["MAX_SEQ_LENGTH"],
                                                              is_training=True, drop_remainder=False)
        if DEBUG:
            print("Beginning Training!")
        current_time = time.time()
        self.estimator.train(input_fn=train_input_fn,
                             max_steps=num_train_steps)
        if DEBUG:
            print("Training took time :", time.time() - current_time,
                  "s, or ", (time.time() - current_time) / 60, "min")

        self.classifier_trained = True

        test_input_fn = run_classifier.input_fn_builder(features=test_features,
                                                        seq_length=self.params["MAX_SEQ_LENGTH"],
                                                        is_training=False, drop_remainder=False)

        # apply model on test set and print all metrics
        if DEBUG:
            print("Evaluating")
        self.estimator.evaluate(input_fn=test_input_fn, steps=None)

    def predict_estimator(self, X, y=None, labels=[0, 1]):
        """
        Predicts output based on training done during former step

        Parameters:
        - X (list or pd.Series):  sentences
        - y (list or pd.Series): labels (if not provided, only the list of predictions is returned)

        Return:
        - list of tuples (sentence, probabilities, predicted label)
        """
        # throw an exception if classifier is not trained
        if not self.classifier_trained:
            raise Exception("Train estimator first")
        if len(X) == 1:  # predict doesn't work if only one element
            X.append("")
        labels = labels
        result = []
        input_features = self.sentences_to_features(
            X, [0 for i in range(len(X))])
        predict_input_fn = run_classifier.input_fn_builder(features=input_features,
                                                           seq_length=self.params["MAX_SEQ_LENGTH"],
                                                           is_training=False, drop_remainder=False)
        predictions = self.estimator.predict(predict_input_fn)

        # output : tuple (sentence, probability of each class, predicted class)
        for sentence, prediction in zip(X, predictions):
            result.append(
                (sentence, np.exp(prediction['probabilities']), labels[prediction['labels']]))

        if y is not None:
            pred = [result[i][2] for i in range(len(result))]
            print("Accuracy: %s" % accuracy_score(y, pred))
            print("Precision: %s" % precision_score(y, pred))
            print("Recall: %s" % recall_score(y, pred))
            print("f1: %s" % f1_score(y, pred))
            print(confusion_matrix(y, pred))

        return result

    def serving_input_fn(self):
        """ Necessary to export the model """
        label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
        input_ids = tf.placeholder(
            tf.int32, [None, self.params["MAX_SEQ_LENGTH"]], name='input_ids')
        input_mask = tf.placeholder(
            tf.int32, [None, self.params["MAX_SEQ_LENGTH"]], name='input_mask')
        segment_ids = tf.placeholder(
            tf.int32, [None, self.params["MAX_SEQ_LENGTH"]], name='segment_ids')
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'label_ids': label_ids,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids})()
        return input_fn

    def export(self, export_path="export"):
        self.estimator._export_to_tpu = False
        self.estimator.export_savedmodel(
            export_path, self.serving_input_fn, as_text=True)


class RangePredict(RangeRaw):
    """
    Class to train the model
    Superclass: RangeRaw

    Attributes :
        - self.classifier_model: instance of ClassifierModel
        - self.input_col: text that will be fed to the model
        - self.output_col: labels of the train sentences
        - self.sequence_to_remove: particular string to remove from the sentences
        - self.seed: for reproductability

    Methods:
        - clean_range: apply standardize_range (from RangeRaw)
        - define_input_output: affects their value to self.input_col and self.output_col
        - train: calls ClassifierModel method train
        - predict_estimator_test:  calls ClassifierModel method predict_estimator
        - export:  calls ClassifierModel method export

    """

    def __init__(self, dataframe,
                 index_sentences, index_labels, language='English',
                 sequence_to_remove=[],
                 seed=42,
                 text_preprocessing=ClassifierModel.TEXT_CLEAN,
                 param_file='default_params.json'):
        RangeRaw.__init__(self, dataframe, index_sentences,
                          index_labels, language)

        self.classifier_model = ClassifierModel()
        self.classifier_model.initialize(
            text_preprocessing=self.text_preprocessing, param_file=param_file)
        self.input_col = None
        self.output_col = None
        self.sequence_to_remove = sequence_to_remove
        self.seed = seed

    def clean_range(self):
        """
        Calls RangeRaw.standardize_range
        """
        self.standardize_range(sequence_to_remove=self.sequence_to_remove)

    def define_input_output(self):
        """
        Defines input column according to model (raw text or clean text)
        Defines output column : labels
        """

        if self.classifier_model.text_preprocessing == ClassifierModel.TEXT_CLEAN:
            self.clean_range()
            self.input_col = self.df["clean_text"]

        elif self.classifier_model.text_preprocessing == ClassifierModel.TEXT_RAW:
            self.input_col = self.df[self.col_name_sentences]

        else:
            raise Exception("Text preprocessing unknown")

        self.output_col = self.df[self.col_name_labels]

    def train(self):
        """
        Call the Classifiermodel method
        """
        if self.input_col is None:
            raise Exception("Preprocessing not specified")
        self.classifier_model.train(self.input_col, self.output_col)

    def predict_estimator_test(self, X, y=None):
        """
        Make predictions for the given dataset
        Warning, the train methods already split the dataset into train and test and evaluate the model on the test set,
        this method should be used only to test the good generalization of the model, on a data set it has never seen
        """
        if self.classifier_model.classifier_trained:
            results = self.classifier_model.predict_estimator(
                X, y, self.classifier_model.label_list)
            return results

    def export(self, export_path="export"):
        self.classifier_model.export(export_path=export_path)

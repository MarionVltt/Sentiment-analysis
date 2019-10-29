# usr/bin/python3
# * encoding=utf_8*

import json
import numpy as np
from preprocessing_en import RangeRaw, SingleRaw
import time
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.saved_model import tag_constants
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import bert  # install with !pip install bert-tensorflow
from bert import run_classifier
from bert import optimization
from bert import tokenization


class ClassifierModel():
    """
    Class that stores all steps from the model to ensure train and test
    are processed the same way
    Shall be instantiated only once
    - self.file_path, index transaction => preprocessing.singleRaw
    - self.test_size : test percentage within dataset
    - self.seed : random seed for sampling

    """
    TEXT_RAW = 0
    TEXT_CLEAN = 1

    # MAX_SEQ_LENGTH = 128
    # BATCH_SIZE = 10
    # LEARNING_RATE = 2e-5
    # NUM_TRAIN_EPOCHS = 3.0
    # # Warmup is a period of time where hte learning rate
    # # is small and gradually increases--usually helps training.
    # WARMUP_PROPORTION = 0.1
    # # Model configs
    # SAVE_CHECKPOINTS_STEPS = 500
    # SAVE_SUMMARY_STEPS = 100
    #
    #BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    BERT_MODEL_HUB = None
    #
    # OUTPUT_DIR = 'output-amazon-test'  # @param {type:"string"}
    # # @markdown Whether or not to clear/delete the directory and create a new one
    # DO_DELETE = False  # @param {type:"boolean"}

    def __init__(self):
        print("Model instantiated")
        self.params = dict()
        self.text_preprocessing = None
        self.algo_name = ""
        #self.algo_accuracy = np.float64(0.0)
        #self.algo_recall = np.float64(0.0)
        #self.algo_precision = np.float64(0.0)
        #self.algo_f1_score = np.float64(0.0)
        self.is_set_up = False
        self.classifier_trained = False
        self.model_path = None
        self.test_size = None
        self.label_list = []
        self.tokenizer = None
        self.estimator = None

    def initialize(self,
                   model_path,
                   text_preprocessing=None,
                   test_size=0.2, param_file='default_params.json'):
        """
        Instance method:
        Its purpose is to:
        - set the output directory of the training
        - create tokenizer


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
        print('***** Model output directory: {} *****'.format(self.params["OUTPUT_DIR"]))

        if not text_preprocessing:
            self.text_preprocessing = ClassifierModel.TEXT_RAW
        else:
            self.text_preprocessing = ClassifierModel.TEXT_CLEAN
        self.is_set_up = True
        self.classifier_trained = False
        self.model_path = model_path
        self.test_size = test_size
        ClassifierModel.BERT_MODEL_HUB = self.params['BERT_MODEL_HUB']
        print("creating tokenizer")
        self.tokenizer = self.create_tokenizer_from_hub_module()
        print("init done")

    def sentences_to_features(self, sentences, labels):
        """ one list with the sentences and one with the corresponding labels
        If prediction : put all labels to 0
        Transform the string into bert features
        """
        input_examples = [run_classifier.InputExample(guid="", text_a=s, text_b=None, label=l) for s, l in
                          zip(sentences, labels)]  # here, "" is just a dummy label
        input_features = run_classifier.convert_examples_to_features(input_examples, self.label_list,
                                                                     self.params["MAX_SEQ_LENGTH"],
                                                                     self.tokenizer)
        return input_features

    def create_tokenizer_from_hub_module(self):
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = hub.Module(self.params["BERT_MODEL_HUB"])
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])
        return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    """The four next methods are necessary for the estimator"""
    @staticmethod
    def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):
        """Creates a classification model."""

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
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return (predicted_labels, log_probs)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, predicted_labels, log_probs)

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    @staticmethod
    def model_fn_builder(num_labels, learning_rate, num_train_steps,
                         num_warmup_steps):
        """Returns `model_fn` closure for TPUEstimator."""

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
        Split X, y as train and test sets. Then convert them to features, classify and score
        Input:
        - X: pd.Series containing text
        - y: pd.Series containing labels

        Shows train and test score to infer whether model is too simple
        or overfitted
        """
        tf.logging.set_verbosity(tf.logging.INFO) #comment if you don't want to display the information during training/evaluation

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42, stratify=y)

        self.label_list = y.unique()

        train_features = self.sentences_to_features(X_train, y_train)
        test_features = self.sentences_to_features(X_test, y_test)
        print("transformation to features done")

        num_train_steps = int(len(train_features) / self.params["BATCH_SIZE"] * self.params["NUM_TRAIN_EPOCHS"])
        num_warmup_steps = int(num_train_steps * self.params["WARMUP_PROPORTION"])

        run_config = self.run_config_builder()
        model_fn = self.model_fn_builder(len(self.label_list), self.params["LEARNING_RATE"], num_train_steps,
                                         num_warmup_steps)
        self.estimator = self.estimator_builder(model_fn, run_config)

        print("creation of all stuff needed done")

        train_input_fn = bert.run_classifier.input_fn_builder(features=train_features,
                                                              seq_length=self.params["MAX_SEQ_LENGTH"],
                                                              is_training=True, drop_remainder=False)

        print(f'Beginning Training!')
        current_time = time.time()
        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("Training took time :", time.time() - current_time, "s, or ", (time.time() - current_time)/60, "min")

        self.classifier_trained = True

        test_input_fn = run_classifier.input_fn_builder(features=test_features,
                                                        seq_length=self.params["MAX_SEQ_LENGTH"],
                                                        is_training=False, drop_remainder=False)

        #apply model on test set and print all metrics
        print("Evaluating")
        self.estimator.evaluate(input_fn=test_input_fn, steps=None)

    def predict_estimator(self, X, y=None, labels=[0, 1]):
        """
        predicts output based on training done during former step
        Input:
        - X: list or pd.Series of sentences
        - y: list or pd.Series of labels (if not provided, only the list of predictions is returned)

        Output:
        - prediction
        """
        # throw an exception if classifier is not trained
        if not self.classifier_trained:
            raise Exception("Train estimator first")
        if len(X) == 1:  # predict doesn't work if only one element
            X.append("")
        labels = labels
        result = []
        input_features = self.sentences_to_features(X, [0 for i in range(len(X))])
        predict_input_fn = run_classifier.input_fn_builder(features=input_features,
                                                           seq_length=self.params["MAX_SEQ_LENGTH"],
                                                           is_training=False, drop_remainder=False)
        predictions = self.estimator.predict(predict_input_fn)

        # output : tuple (sentence, probability of each class, predicted class)
        for sentence, prediction in zip(X, predictions):
            result.append((sentence, np.exp(prediction['probabilities']), labels[prediction['labels']]))

        if y is not None:
            pred = [result[i][2] for i in range(len(result))]
            print("Accuracy: %s" % accuracy_score(y, pred))
            print("Precision: %s" % precision_score(y, pred))
            print("Recall: %s" % recall_score(y, pred))
            print("f1: %s" % f1_score(y, pred))
            print(confusion_matrix(y, pred))

        return result

    """ necessary to export the model"""
    def serving_input_fn(self):
        label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
        input_ids = tf.placeholder(tf.int32, [None, self.params["MAX_SEQ_LENGTH"]], name='input_ids')
        input_mask = tf.placeholder(tf.int32, [None, self.params["MAX_SEQ_LENGTH"]], name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [None, self.params["MAX_SEQ_LENGTH"]], name='segment_ids')
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'label_ids': label_ids,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        })()
        return input_fn

    def export(self, export_path="export"):
        self.estimator._export_to_tpu = False
        self.estimator.export_savedmodel(export_path, self.serving_input_fn, as_text=True)


class RangePredict(RangeRaw):
    """
    Class to train the model:
    - Superclass: RangeRaw
    - Component: ClassifierModel

    """

    def __init__(self, dataframe,
                 index_sentences, index_labels,
                 sequence_to_remove=[],
                 seed=42,
                 model_path="model_dir",
                 text_preprocessing=ClassifierModel.TEXT_CLEAN,
                 test_size=0.2, param_file='default_params.json'):
        RangeRaw.__init__(self, dataframe, index_sentences, index_labels)

        self.input_col = None
        self.output_col = None
        self.sequence_to_remove = sequence_to_remove
        self.seed = seed
        self.model_path = model_path
        self.text_preprocessing = text_preprocessing
        self.test_size = test_size
        # self.tensor_input_ids = None
        # self.tensor_input_mask = None
        # self.tensor_label_ids = None
        # self.tensor_segment_ids = None
        # self.tensor_outputs = None
        # self.sess = tf.Session()
        self.classifier_model = ClassifierModel()
        self.classifier_model.initialize(self.model_path,
                                         text_preprocessing=self.text_preprocessing,
                                         test_size=self.test_size, param_file=param_file)

    def clean_range(self):
        """
        - Calls RangeRaw.standardize_range
        """
        self.standardize_range(sequence_to_remove=self.sequence_to_remove)

    def preprocess(self):
        """
        Use either raw text or clean text
        depending on the model
                - Defines input column according to model
                - Defines output column : labels
        """

        if self.classifier_model.text_preprocessing == ClassifierModel.TEXT_CLEAN:
            self.clean_range()
            self.input_col = self.df["clean_text"]

        elif self.classifier_model.text_preprocessing == ClassifierModel.TEXT_RAW:
            self.input_col = self.df[self.col_name_sentences]

        else:
            raise Exception("Text preprocessing unknown")

        self.output_col = self.df[self.col_name_labels]
        print("preprocess done")

    """Call the Classifiermodel methods"""
    def train(self):
        if self.input_col is None:
            raise Exception("Preprocessing not specified")
        self.classifier_model.train(self.input_col, self.output_col)

    def predict_estimator_test(self, X, y=None):
        """ Make predictions for the given dataset
            Warning, the train methods already split the dataset into train and test and evaluate the model on the test set,
            this method should be used only to test the good generalization of the model, on a data set it has never seen
        """
        if self.classifier_model.classifier_trained:
            results = self.classifier_model.predict_estimator(X, y, self.classifier_model.label_list)
            return results

    def export(self, export_path="export"):
        self.classifier_model.export(export_path=export_path)


class SinglePredict(SingleRaw):
    """
    Class to predict the category:
    Superclass: SingleRaw
    Component: ClassifierModel
    """
    # class attributes
    classifier_model_loaded = False
    # definition of the tensorflow graph
    tensor_input_ids = None
    tensor_input_mask = None
    tensor_label_ids = None
    tensor_segment_ids = None
    tensor_outputs = None
    sess = None
    # needed for bert
    BERT_MODEL_HUB = None
    tokenizer = None
    # needed for first prediction
    initialized = False
    initializing_raw = "Hi, I'm here to run the model a first time before the users comes in to make his predictions faster."

    def __init__(self, sentence, label_list=[0, 1], sequence_to_remove=[], max_seq_length=128, model_path="model_dir"):

        SingleRaw.__init__(self, sentence)
        self.model_path = model_path
        self.predicted_class = None
        # self.predicted_proba = None
        # self.predicted_log_proba = None
        self.input_features = None
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.empty = False
        self.sequence_to_remove = sequence_to_remove
        # in order to load model only once
        if not SinglePredict.classifier_model_loaded:
            SinglePredict.sess = tf.Session()
            t0 = time.time()
            SinglePredict.BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
            SinglePredict.tokenizer = SinglePredict.create_tokenizer_from_hub_module()
            t1 = time.time()
            SinglePredict.import_model(model_path)
            t2 = time.time()
            print("Model loaded : \n time bert: {}, time import: {}".format(t1-t0, t2-t1))

    @staticmethod
    def initialize(model_path="model_dir"):
        """ Load the model and run it on a first sentence"""
        t2bis = time.time()
        init = SinglePredict(SinglePredict.initializing_raw, model_path=model_path)
        t3bis = time.time()
        init.preprocess()
        init.predict()
        t3 = time.time()
        print("time init complete: {}, time init sentence: {}".format(t3-t2bis, t3-t3bis))

    @staticmethod
    def create_tokenizer_from_hub_module():
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = hub.Module(SinglePredict.BERT_MODEL_HUB)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])
        return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    @staticmethod
    def import_model(dir_path):
        """ Import tensorflow model and store the tensors"""
        export_dir = dir_path

        tf.saved_model.loader.load(SinglePredict.sess, [tag_constants.SERVING], export_dir)
        SinglePredict.tensor_input_ids = SinglePredict.sess.graph.get_tensor_by_name('input_ids_1:0')
        SinglePredict.tensor_input_mask = SinglePredict.sess.graph.get_tensor_by_name('input_mask_1:0')
        SinglePredict.tensor_label_ids = SinglePredict.sess.graph.get_tensor_by_name('label_ids_1:0')
        SinglePredict.tensor_segment_ids = SinglePredict.sess.graph.get_tensor_by_name('segment_ids_1:0')
        SinglePredict.tensor_outputs = SinglePredict.sess.graph.get_tensor_by_name('loss/Squeeze:0')
        SinglePredict.classifier_model_loaded = True

    def preprocess(self):
        """Method from preprocessing_en.py"""
        self.standardize_raw(self.sequence_to_remove)
        if len(self.clean_text.split()) == 0:
            self.empty = True
        # print("preprocess : ", self.clean_text)

    def predict(self):
        if not self.empty:
            """ See runclassifier.py from the bert git for more details"""
            input_examples = [run_classifier.InputExample(guid="", text_a=self.clean_text, text_b=None,
                                                          label=self.label_list[0])]  # here, "" is just a dummy label
            self.input_features = run_classifier.convert_examples_to_features(input_examples, self.label_list,
                                                                              self.max_seq_length, self.tokenizer)
            #assert len(self.input_features) == 1
            t0 = time.time()
            input_ids = np.array(self.input_features[0].input_ids).reshape(-1, self.max_seq_length)
            t1 = time.time()
            input_mask = np.array(self.input_features[0].input_mask).reshape(-1, self.max_seq_length)
            t2 = time.time()
            label_ids = np.array(self.input_features[0].label_id).reshape(-1, )
            t3 = time.time()
            segment_ids = np.array(self.input_features[0].segment_ids).reshape(-1, self.max_seq_length)
            t4 = time.time()

            result = SinglePredict.sess.run(SinglePredict.tensor_outputs, feed_dict={
                SinglePredict.tensor_input_ids: input_ids,
                SinglePredict.tensor_input_mask: input_mask,
                SinglePredict.tensor_label_ids: label_ids,
                SinglePredict.tensor_segment_ids: segment_ids,
            })
            t5 = time.time()
            self.predicted_class = self.label_list[result]
            t6 = time.time()
            return t0, t1, t2, t3, t4, t5, t6
        else:
            return 0, 0, 0, 0, 0, 0, 0

    def clean_and_predict(self):
        self.preprocess()
        self.predict()
        print("The predicted class is: {}".format(self.predicted_class))

    @staticmethod
    def unload():
        SinglePredict.sess.close()

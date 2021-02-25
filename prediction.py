# usr/bin/python3

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.saved_model import tag_constants

import bert  # install with pip install bert-tensorflow
from bert import run_classifier
from bert import tokenization

from preprocessing import RangeRaw, SingleRaw


class SinglePredict(SingleRaw):
    """
    Class to predict the category (positive or negative) of a sentence
    Superclass: SingleRaw

    Attributes:
        - self.model_path: path to the trained model
        - self.predicted_class: will store the prediction for the sentence
        - self.input_features: BERT input features
        - self.max_seq_length: number maximum of characters in the sentence
        - self.empty : True is the cleaned sequence is empty
        - self.sequence_to_remove: particular string to remove from the sentence

    Methods:
        - initialize (static): loads the model and run it on a first sentence
        - import_model (static): imports the tensorflow stored model
        - preprocess: calls standardize_raw from SingleRaw
        - predict
        - clean_and_predict: combination of preprocess and predict
        - unload: closes the tensorflow session
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

    def __init__(self, sentence, language='English', label_list=[0, 1], sequence_to_remove=[], max_seq_length=128, model_path="model_dir"):

        SingleRaw.__init__(self, sentence, language)
        self.model_path = model_path
        self.predicted_class = None
        self.input_features = None
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.empty = False
        self.sequence_to_remove = sequence_to_remove
        # in order to load model only once when used in a production environment
        if not SinglePredict.classifier_model_loaded:
            SinglePredict.sess = tf.Session()
            SinglePredict.BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
            SinglePredict.tokenizer = SinglePredict.create_tokenizer_from_hub_module()
            SinglePredict.import_model(model_path)
            print("Model loaded")

    @staticmethod
    def initialize(model_path="model_dir"):
        """
        To run before the first prediction
        Load the model and run it on a first sentence (the first prediction takes longer)"""
        init = SinglePredict(
            SinglePredict.initializing_raw, model_path=model_path)
        init.preprocess()
        init.predict()
        print("Model initialized")

    @ staticmethod
    def create_tokenizer_from_hub_module():
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = hub.Module(SinglePredict.BERT_MODEL_HUB)
            tokenization_info = bert_module(
                signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])
        return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    @ staticmethod
    def import_model(dir_path):
        """ Import tensorflow model and store the tensors"""
        export_dir = dir_path

        tf.saved_model.loader.load(
            SinglePredict.sess, [tag_constants.SERVING], export_dir)
        SinglePredict.tensor_input_ids = SinglePredict.sess.graph.get_tensor_by_name(
            'input_ids_1:0')
        SinglePredict.tensor_input_mask = SinglePredict.sess.graph.get_tensor_by_name(
            'input_mask_1:0')
        SinglePredict.tensor_label_ids = SinglePredict.sess.graph.get_tensor_by_name(
            'label_ids_1:0')
        SinglePredict.tensor_segment_ids = SinglePredict.sess.graph.get_tensor_by_name(
            'segment_ids_1:0')
        SinglePredict.tensor_outputs = SinglePredict.sess.graph.get_tensor_by_name(
            'loss/Squeeze:0')
        SinglePredict.classifier_model_loaded = True

    def preprocess(self):
        """
        Calls standardize_raw from preprocessing.py
        """
        self.standardize_raw(self.sequence_to_remove)
        if len(self.clean_text.split()) == 0:
            self.empty = True

    def predict(self):
        """
        Predict and store the predicted label in self.predicted_class
        """
        if not self.empty:
            """ See runclassifier.py from the bert git for more details"""
            input_examples = [run_classifier.InputExample(guid="", text_a=self.clean_text, text_b=None,
                                                          label=self.label_list[0])]  # here, "" is just a dummy label
            self.input_features = run_classifier.convert_examples_to_features(input_examples, self.label_list,
                                                                              self.max_seq_length, self.tokenizer)
            input_ids = np.array(
                self.input_features[0].input_ids).reshape(-1, self.max_seq_length)
            input_mask = np.array(
                self.input_features[0].input_mask).reshape(-1, self.max_seq_length)
            label_ids = np.array(self.input_features[0].label_id).reshape(-1, )
            segment_ids = np.array(
                self.input_features[0].segment_ids).reshape(-1, self.max_seq_length)

            result = SinglePredict.sess.run(SinglePredict.tensor_outputs, feed_dict={
                SinglePredict.tensor_input_ids: input_ids,
                SinglePredict.tensor_input_mask: input_mask,
                SinglePredict.tensor_label_ids: label_ids,
                SinglePredict.tensor_segment_ids: segment_ids})
            self.predicted_class = self.label_list[result]

    def clean_and_predict(self):
        self.preprocess()
        self.predict()
        print("The predicted class is: {}".format(self.predicted_class))

    @staticmethod
    def unload():
        SinglePredict.sess.close()

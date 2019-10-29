import numpy as np
import json
from flask import Flask, request, jsonify
application = Flask(__name__)
import requests
import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import run_classifier


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])
    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


tokenizer = create_tokenizer_from_hub_module()


@application.route('/ping', methods=['GET'])
def pong():
    return "pong"


@application.route('/sentiment_en', methods=['POST'])
def sentiment_en():
    req = request.get_json()
    # print(req)
    raw = req["raw"]
    label_list = [0, 1]
    input_examples = [run_classifier.InputExample(guid="", text_a=raw, text_b=None,
                                                  label=0)]  # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, 128, tokenizer)
    input_ids = np.array(input_features[0].input_ids)
    segment_ids = np.array(input_features[0].segment_ids)
    input_masks = np.array(input_features[0].input_mask)
    label_ids = input_features[0].label_id
    tensor_dict = {"label_ids": [label_ids], "segment_ids": [segment_ids.tolist()],
                   "input_mask": [input_masks.tolist()], "input_ids": [input_ids.tolist()]}
    data_dict = {"inputs": tensor_dict}
    data = json.dumps(data_dict)
    response = requests.post("http://localhost:8501/v1/models/model_en:predict ", data=data)
    print(response)
    response = response.json()
    print(response)
    return jsonify(response)


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=2121)

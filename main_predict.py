from flask import Flask, request, jsonify
application = Flask(__name__)
import time
import prediction_en, prediction_fr
print('starting english model')
s = time.time()
prediction_en.SinglePredict.initialize(model_path="model_dir")
print('end in: ', time.time() - s)
print('starting french model')
s = time.time()
prediction_fr.SinglePredict.initialize(model_path="model_dir_fr")
print('end in: ', time.time() - s)

@application.route('/ping', methods=['GET'])
def pong():
    return "pong"

@application.route('/sentiment_fr', methods=['POST'])
def return_sentiment_french():
    req = request.get_json()
    if req["processus"] and "raw" in req["processus"]:
        raw = req["processus"]["raw"]
    else:
        return {"error": "MISSING_PARAMETERS", 'note': 'return_sentiment_french need a string argument'}
    if not isinstance(raw, str):
        return {"error": "MISSING_PARAMETERS", 'note': 'return_sentiment_french need a string argument'}
    new_predict = prediction_fr.SinglePredict(raw, ["Neg", "Pos"])
    new_predict.clean_and_predict()
    sentiment = new_predict.predicted_class
    return jsonify(sentiment)

@application.route('/sentiment_en', methods=['POST'])
def return_sentiment_english():
    req = request.get_json()
    if req["processus"] and "raw" in req["processus"]:
        raw = req["processus"]["raw"]
    else:
        return {"error": "MISSING_PARAMETERS", 'note': 'return_sentiment_english need a string argument'}
    if not isinstance(raw, str):
        return {"error": "MISSING_PARAMETERS", 'note': 'return_sentiment_english need a string argument'}
    new_predict = prediction_en.SinglePredict(raw, ["Neg", "Pos"])
    new_predict.clean_and_predict()
    sentiment = new_predict.predicted_class
    return jsonify(sentiment)

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=2121)

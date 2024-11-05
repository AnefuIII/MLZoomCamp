
import pickle
from flask import Flask
from flask import request
from flask import jsonify


model = "model1.bin"
dv = "dv.bin"

with open(model, "rb") as f_in:
    model = pickle.load(f_in)

with open(dv, "rb") as f_in2:
    dv = pickle.load(f_in2)


app = Flask('client')

@app.route('/predict', methods = ['POST']) #use post method because we want to send information
def predict():
    '''how to tell the app the request/data of the 
    customer we want to predics will come as a json format'''
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    sub = y_pred >= 0.5

    result = {
        "Churn Probability": float(y_pred),
        "Churn": bool(sub)
    }

    return jsonify(result) # to output a json file


if __name__== '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)



from flask import Flask
from flask import request
from flask import jsonify
import pickle

app = Flask("heart_disease")

output_file = "model_rf_est11_depth15.bin"

with open(output_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

@app.route('/predict', methods = ['POST'])
def predict(): 
    # get_json will convert the body of the request to a python dictionary
    patient = request.get_json()

    X_pred = dv.transform([patient])

    pred = model.predict_proba(X_pred)[0,1]
    risk = pred >= 0.5

    result = {"The patient heart disease probability is: ": float(pred),
              "The risk is ": bool(risk)}

    #converts the result to go back as a json format
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)


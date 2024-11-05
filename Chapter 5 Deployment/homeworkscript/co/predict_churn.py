import pickle


file_output = "model_C=1.0.bin"

with open(file_output, "rb") as f_in:
    dv, model = pickle.load(f_in)


customer = {"customerid": "9305-cdskc",
 "gender": "female",
 "seniorcitizen": 0,
 "partner": "no",
 "dependents": "no",
 "tenure": 8,
 "phoneservice": "yes",
 "multiplelines": "yes",
 "internetservice": "fiber_optic",
 "onlinesecurity": "no",
 "onlinebackup": "no",
 "deviceprotection": "yes",
 "techsupport": "no",
 "streamingtv": "yes",
 "streamingmovies": "yes",
 "contract": "month-to-month",
 "paperlessbilling": "yes",
 "paymentmethod": "electronic_check",
 "monthlycharges": 99.65,
 "totalcharges": 820.5}

X = dv.transform(customer)

y_pred = model.predict_proba(X)[0, 1]

print("input: ", customer)
print("churn probability: ", y_pred)
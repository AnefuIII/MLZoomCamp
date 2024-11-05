import pickle


model = "model1.bin"
dv = "dv.bin"

with open(model, "rb") as f_in:
    model = pickle.load(f_in)

with open(dv, "rb") as f_in2:
    dv = pickle.load(f_in2)


client = {"job": "student", "duration": 280, "poutcome": "failure"}

X = dv.transform(client)

y_pred = model.predict_proba(X)[0, 1]

print("input: ", client)
print("churn probability: ", y_pred)
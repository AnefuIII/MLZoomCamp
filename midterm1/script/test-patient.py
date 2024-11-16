import requests

url = 'http://localhost:9696/predict'

patient = {"age": 56,
 "sex": 1,
 "cp": 1,
 "trestbps": "120",
 "chol": "236",
 "fbs": "0",
 "restecg": "1",
 "thalachh": "178",
 "exang": "0",
 "oldpeak": 0.8,
 "slope": "2",
 "ca": "0",
 "thal": "2",
  }

response = requests.post(url, json = patient).json()
print(response)


#!/usr/bin/env python
# coding: utf-8


#get_ipython().run_line_magic('autosave', '0')



import requests



url = "http://localhost:9696/predict"


customer = {
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
 "techsupport": "yes",
 "streamingtv": "yes",
 "streamingmovies": "yes",
 "contract": "month-to-month",
 "paperlessbilling": "yes",
 "paymentmethod": "electronic_check",
 "monthlycharges": 19.65,
 "totalcharges": 200.5}



response = requests.post(url, json = customer).json()
print(response)




if response['Churn'] == True:
    print('sending email to customer %s' % ('9305-cdskc'))





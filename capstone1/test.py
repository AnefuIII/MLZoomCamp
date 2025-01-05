import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {"url": "CIFAKE/final_test/FAKE/990 (2).jpg"}
# data_full = {'url': 'C:/Users/HP/Desktop/MLZOOMCAMP/notebooks/capstone1/CIFAKE/final_test/FAKE/990 (2).jpg'}

#data = {'url': 'https://piktochart.com/wp-content/uploads/2023/11/pope-wearing-balenciaga-jacket-viral-ai-images.jpeg'}

result = requests.post(url, json = data).json()
print(result)
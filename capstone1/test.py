import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

#data = {"url": "CIFAKE/final_test/FAKE/990 (2).jpg"} #local machine

#data = {"url": "/var/task/CIFAKE/final_test/FAKE/990 (2).jpg"} #volume

data = {'url': 'https://plus.unsplash.com/premium_photo-1664303218668-03fa4e612038?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHwxfHx8ZW58MHx8fHx8'}

result = requests.post(url, json = data).json()
print(result)



# import requests
# import base64

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# with open("C:/Users/HP/Desktop/MLZOOMCAMP/notebooks/capstone1/CIFAKE/final_test/FAKE/990 (2).jpg", "rb") as f:
#     image_data = f.read()
#     base64_encoded_image = base64.b64encode(image_data).decode('utf-8')

# data = {"image_data": base64_encoded_image}

# result = requests.post(url, json=data).json()
# print(result)
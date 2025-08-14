import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "Breaking news: AI is taking over the world!"}

response = requests.post(url, json=data)
print(response.json())

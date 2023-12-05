import requests

r=requests.post('http://127.0.0.1:5000/iris_predict',json={"mydata":'[[5.1],[3.5],[1.4],[0.2]]'})

r.status_code
r.text

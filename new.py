from flask import Flask,request, Response
from joblib import load
import numpy as np

# Load the pre-trained logistic regression model
my_lr_model=load("model/iris_prediction.joblib")
#Initializing
app = Flask(__name__)

#Creating the very first route


@app.route("/iris_predict",methods=['POST','GET'])
def iris_predict():
    data=request.json
 # Convert user input data to a NumPy array and reshape it to match the model's input format
    user_sent_this_data=data.get('mydata')
    user_number=np.array(user_sent_this_data).reshape(1, -1)

    #using the users data and giving it to our model
    model_prediction=my_lr_model.predict(user_number)

    #returning the response
    return Response(str(model_prediction))

if __name__=='__main__':
    app.run(debug=True)


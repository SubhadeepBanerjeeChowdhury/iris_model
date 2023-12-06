# iris_model
Flask API for Iris Flower Prediction Overview:
This Flask application establishes an API endpoint (/iris_predict) to receive user input, apply a pre-trained Logistic Regression Classification model to generate predictions based on the provided data.

Model Loading:
The ML model has already been created and saved in the form of a ‘joblib’ file. The script initiates by loading a pre-trained machine learning model file from the designated file path: 'model/iris_prediction.joblib'

API Endpoint ("/iris_predict"):

Defining a singular API endpoint, "/iris_predict," the Flask application is equipped to handle both POST and GET requests. This endpoint acts as the primary portal for users to submit data, triggering predictions.

Prediction Function:
Upon receiving a request, the script efficiently extracts user-provided data embedded in the JSON payload.
The extracted data undergoes transformation into a NumPy array, subsequently reshaped to align with the anticipated input structure of the logistic regression model.
Leveraging the pre-trained model, predictions regarding iris flower types are made based on the user's input.

Response:
The model's predictions are converted into a string format, serving as the response to fulfill the user's request. In response to the user’s input data, the name of the predicted flower species is returned.

App Execution:

When the script is executed directly, the Flask application operates in debug mode, streamlining the testing and development process.

Summary:

In essence, this Flask API offers an accessible interface for users to engage with a pre-trained logistic regression model, facilitating predictions for iris flower types. Submitting data to the "/iris_predict" endpoint enables users to receive predictions, making it a valuable tool for integrating machine learning capabilities into diverse applications. Additionally, the application demonstrates a commitment to robust functionality by incorporating basic error handling. During deployment, the Flask app's debug mode further simplifies the development and testing phases.

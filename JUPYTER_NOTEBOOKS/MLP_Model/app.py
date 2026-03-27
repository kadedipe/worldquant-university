from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model_filename = "final_mlp_model.pkl"
model = joblib.load(model_filename)

# Create a Flask app
app = Flask(__name__)

# Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    input_data = request.json

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make a prediction using the model
    prediction = model.predict(input_df)

    # Return the prediction as a JSON response
    return jsonify({"prediction": int(prediction[0])})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

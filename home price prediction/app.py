from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        sqft_living = float(request.form['sqft_living'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        floors = int(request.form['floors'])

        # Create feature vector
        features = [[sqft_living, bedrooms, bathrooms, floors]]

        # Predict house price
        prediction = model.predict(features)

        # Return result page
        return render_template('result.html', price=round(prediction[0], 2))
    except Exception as e:
        return f"An error occurred: {e}"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

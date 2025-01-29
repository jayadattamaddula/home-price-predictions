import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv('kc_house_data.csv')

# Select features and target variable
X = data[['sqft_living', 'bedrooms', 'bathrooms', 'floors']]
y = data['price']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'house_price_model.pkl')
print("Model trained and saved as house_price_model.pkl.")

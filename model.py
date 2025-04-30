import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)



# Load the dataset
data = pd.read_csv('./data.csv')

# Remove any unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Ensure 'diagnosis' is mapped correctly
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})

# Verify feature count
num_features = X.shape[1]
print(f"Number of features: {num_features}")

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and calculate accuracy
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"RandomForest Model Accuracy: {rf_accuracy:.2f}")

# Train a Neural Network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(num_features,)),  # Ensure correct input size
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=1)

# Evaluate the model
nn_loss, nn_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Model Accuracy: {nn_accuracy:.2f}")

# API for predicting
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data', [])
        
        if len(data) != num_features:
            return jsonify({'error': f'Expected {num_features} features, got {len(data)}'})

        data = np.array(data).reshape(1, -1)
        feature_names = X.columns
        data_df = pd.DataFrame(data, columns=feature_names)

        # Preprocess and predict
        data_scaled = scaler.transform(data_df)
        prediction_prob = model.predict(data_scaled)[0][0]
        prediction = 1 if prediction_prob >= 0.5 else 0

        confidence = prediction_prob * 100 if prediction == 1 else (1 - prediction_prob) * 100

        result = {
            'prediction': int(prediction),
            'confidence': f"{confidence:.2f}%"
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

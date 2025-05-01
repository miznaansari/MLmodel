import os
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS

# Disable GPU to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Enable mixed precision to reduce memory usage
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load the dataset
try:
    data = pd.read_csv('./data.csv')
    logger.info("Dataset loaded successfully")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Remove any unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Ensure 'diagnosis' is mapped correctly
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})

# Verify feature count
num_features = X.shape[1]
logger.info(f"Number of features: {num_features}")

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
logger.info(f"RandomForest Model Accuracy: {rf_accuracy:.2f}")

# Train a smaller Neural Network to reduce memory usage
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')  # Ensure output is float32 for mixed precision
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with smaller batch size
try:
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=1)
    logger.info("Neural network training completed")
except Exception as e:
    logger.error(f"Error training neural network: {e}")
    raise

# Evaluate the model
nn_loss, nn_accuracy = model.evaluate(X_test_scaled, y_test)
logger.info(f"Neural Network Model Accuracy: {nn_accuracy:.2f}")

# API for predicting
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data', [])
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if len(data) != num_features:
            return jsonify({'error': f'Expected {num_features} features, got {len(data)}'}), 400

        data = np.array(data, dtype=np.float32).reshape(1, -1)
        feature_names = X.columns
        data_df = pd.DataFrame(data, columns=feature_names)

        # Preprocess and predict
        data_scaled = scaler.transform(data_df)
        prediction_prob = model.predict(data_scaled, verbose=0)[0][0]
        prediction = 1 if prediction_prob >= 0.5 else 0
        confidence = prediction_prob * 100 if prediction == 1 else (1 - prediction_prob) * 100

        result = {
            'prediction': int(prediction),
            'confidence': f"{confidence:.2f}%"
        }
        
        logger.info(f"Prediction made: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use Render's PORT environment variable or default to 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
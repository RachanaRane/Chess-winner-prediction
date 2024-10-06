from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

app = Flask(__name__)

# Load the trained XGBoost model
xgb_model = joblib.load('models/xgb_model.pkl')

# Load label encoders
label_encoders = joblib.load('models/labelencoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    created_at = float(request.form['created_at'])
    last_move_at = float(request.form['last_move_at'])
    turns = int(request.form['turns'])
    victory_status = request.form['victory_status']
    increment_code = request.form['increment_code']
    white_rating = int(request.form['white_rating'])
    black_rating = int(request.form['black_rating'])
    opening_eco = request.form['opening_eco']
    opening_name = request.form['opening_name']
    opening_ply = int(request.form['opening_ply'])

    # Create input data dictionary
    input_data = {
        'created_at': [created_at],
        'last_move_at': [last_move_at],
        'turns': [turns],
        'victory_status': [victory_status],
        'increment_code': [increment_code],
        'white_rating': [white_rating],
        'black_rating': [black_rating],
        'opening_eco': [opening_eco],
        'opening_name': [opening_name],
        'opening_ply': [opening_ply]
    }

    # Preprocess input data and predict winner
    predicted_winner = predict_winner(input_data)
    return render_template('result.html', predicted_winner=predicted_winner)

def predict_winner(input_data):
    # Preprocess the input data
    input_df = pd.DataFrame(input_data)
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Make predictions using the trained model
    prediction = xgb_model.predict(input_df)
    predicted_winner = label_encoders['winner'].inverse_transform(prediction)[0]
    return predicted_winner

if __name__ == '__main__':
    app.run(debug=True)

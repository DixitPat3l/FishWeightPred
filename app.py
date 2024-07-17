from flask import Flask, request, jsonify, send_file, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the models, scaler, and label encoder
reg_model = joblib.load('linear_regression_model.pkl')
log_reg_model = joblib.load('logistic_regression_model.pkl')
clf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Load the dataset
file_path = r'C:\Users\dixit\DURHAM\ArtificialIntelligence\Semester 2\AIDI 2004\Week 12\Lab4\data\Fish.csv'
fish_data = pd.read_csv(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training_data.html')
def training_data():
    return render_template('training_data.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    inputs = np.array([data['Length1'], data['Length2'], data['Length3'], data['Height'], data['Width']]).reshape(1, -1)
    inputs_scaled = scaler.transform(inputs)
    
    # Predict weight using Linear Regression
    weight_prediction = reg_model.predict(inputs_scaled)[0]
    
    # Predict species using Logistic Regression
    species_logistic_prediction = log_reg_model.predict(inputs_scaled)[0]
    species_logistic = le.inverse_transform([species_logistic_prediction])[0]
    
    # Predict species using Random Forest
    species_rf_prediction = clf_model.predict(inputs_scaled)[0]
    species_rf = le.inverse_transform([species_rf_prediction])[0]
    
    return jsonify({
        'predicted_weight': weight_prediction,
        'predicted_species_logistic': species_logistic,
        'predicted_species_rf': species_rf
    })

@app.route('/descriptive_stats', methods=['GET'])
def descriptive_stats():
    stats = fish_data.describe().to_dict()
    return jsonify(stats)

@app.route('/box_plot_weight.png', methods=['GET'])
def box_plot_weight():
    return send_file('box_plot_weight.png', mimetype='image/png')

@app.route('/box_plots_other.png', methods=['GET'])
def box_plots_other():
    return send_file('box_plots_other.png', mimetype='image/png')

@app.route('/histograms.png', methods=['GET'])
def histograms():
    return send_file('histograms.png', mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
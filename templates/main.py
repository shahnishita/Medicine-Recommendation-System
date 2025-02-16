import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__, template_folder=os.getenv('TEMPLATES_PATH', 'C:/Users/DELL/Desktop/ML projects/medicine recommendation system/templates'))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'nishu*2003')

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', 'C:/Users/DELL/Desktop/ML projects/medicine recommendation system/templates/svc.pkl')
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Load CSV data
SYMPTOMS_CSV = os.getenv('SYMPTOMS_CSV', 'C:/Users/DELL/Desktop/ML projects/medicine recommendation system/symtoms_df.csv')
PRECAUTIONS_CSV = os.getenv('PRECAUTIONS_CSV', 'C:/Users/DELL/Desktop/ML projects/medicine recommendation system/precautions_df.csv')
WORKOUT_CSV = os.getenv('WORKOUT_CSV', 'C:/Users/DELL/Desktop/ML projects/medicine recommendation system/workout_df.csv')
DESCRIPTION_CSV = os.getenv('DESCRIPTION_CSV', 'C:/Users/DELL/Desktop/ML projects/medicine recommendation system/description.csv')
MEDICATIONS_CSV = os.getenv('MEDICATIONS_CSV', 'C:/Users/DELL/Desktop/ML projects/medicine recommendation system/medications.csv')
DIETS_CSV = os.getenv('DIETS_CSV', 'C:/Users/DELL/Desktop/ML projects/medicine recommendation system/diets.csv')

symptoms_df = pd.read_csv(SYMPTOMS_CSV)
precautions_df = pd.read_csv(PRECAUTIONS_CSV)
workout_df = pd.read_csv(WORKOUT_CSV)
description_df = pd.read_csv(DESCRIPTION_CSV)
medications_df = pd.read_csv(MEDICATIONS_CSV)
diets_df = pd.read_csv(DIETS_CSV)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json.get('symptoms', [])
    prediction = model.predict([symptoms])
    disease = prediction[0]

    response = {
        'disease': disease,
        'description': description_df[description_df['Disease'] == disease]['Description'].values[0],
        'precautions': precautions_df[precautions_df['Disease'] == disease]['Precaution'].tolist(),
        'workout': workout_df[workout_df['Disease'] == disease]['Workout'].tolist(),
        'medications': medications_df[medications_df['Disease'] == disease]['Medication'].tolist(),
        'diets': diets_df[diets_df['Disease'] == disease]['Diet'].tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=(os.getenv('FLASK_ENV', 'development') == 'development'))

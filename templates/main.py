import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # Enable CORS for frontend access

# Initialize Flask App
app = Flask(__name__, template_folder='templates')

# Enable CORS for frontend URL
CORS(app, resources={r"/*": {"origins": "https://medicine-recommendation-system-x2wb-ggczsj9fq.vercel.app"}})

# Load datasets (Ensure files are in the project directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sym_des = pd.read_csv(os.path.join(BASE_DIR, 'data/symptoms_df.csv'))
precautions = pd.read_csv(os.path.join(BASE_DIR, 'data/precautions_df.csv'))
workout = pd.read_csv(os.path.join(BASE_DIR, 'data/workout_df.csv'))
description = pd.read_csv(os.path.join(BASE_DIR, 'data/description.csv'))
medications = pd.read_csv(os.path.join(BASE_DIR, 'data/medications.csv'))
diets = pd.read_csv(os.path.join(BASE_DIR, 'data/diets.csv'))

# Load ML Model
model_path = os.path.join(BASE_DIR, 'models/svc.pkl')
svc = pickle.load(open(model_path, 'rb'))

# Helper Function
def helper(disease):
    desc = description[description['Disease'] == disease]['Description'].values
    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    med = medications[medications['Disease'] == disease]['Medication'].values
    diet = diets[diets['Disease'] == disease]['Diet'].values
    wrkout = workout[workout['disease'] == disease]['workout'].values

    return (desc[0] if len(desc) > 0 else "No description available",
            pre.tolist() if len(pre) > 0 else [],
            med.tolist() if len(med) > 0 else [],
            diet.tolist() if len(diet) > 0 else [],
            wrkout.tolist() if len(wrkout) > 0 else [])

# Dictionary of Symptoms & Diseases
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8
}
diseases_list = {0: 'Fungal infection', 1: 'AIDS', 2: 'Acne', 3: 'Alcoholic hepatitis'}

# Model Prediction Function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
        else:
            return None  # Handle unknown symptoms
    return diseases_list.get(svc.predict([input_vector])[0], "Unknown Disease")

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms')

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    user_symptoms = [s.strip().lower() for s in symptoms.split(',') if s.strip().lower() in symptoms_dict]

    if not user_symptoms:
        return jsonify({"error": "Invalid or misspelled symptoms"}), 400

    predicted_disease = get_predicted_value(user_symptoms)
    if not predicted_disease:
        return jsonify({"error": "No matching disease found"}), 400

    desc, precautions, medications, diet, workout = helper(predicted_disease)

    return jsonify({
        "predicted_disease": predicted_disease,
        "description": desc,
        "precautions": precautions,
        "medications": medications,
        "diet": diet,
        "workout": workout
    })

# About & Other Pages
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

# Start Flask App
if __name__ == '__main__':
    app.run(debug=True)

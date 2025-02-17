import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load datasets
sym_des = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/Final Year Project/flask_env/symtoms_df.csv')
precautions = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/Final Year Project/flask_env/precautions_df.csv')
workout = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/Final Year Project/flask_env/workout_df.csv')
description = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/Final Year Project/flask_env/description.csv')
medications = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/Final Year Project/flask_env/medications.csv')
diets = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/Final Year Project/flask_env/diets.csv')

# Load the model
svc = pickle.load(open('C:/Users/DELL/OneDrive/Desktop/Final Year Project/flask_env/svc.pkl', 'rb'))

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values[0]
    
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten().tolist()
    
    med = medications[medications['Disease'] == dis]['Medication'].tolist()
    
    die = diets[diets['Disease'] == dis]['Diet'].tolist()
    
    wrkout = workout[workout['disease'] == dis]['workout'].tolist()
    
    return desc, pre, med, die, wrkout

# Symptoms and Disease mappings
symptoms_dict = {  # Your symptoms dictionary remains unchanged }
diseases_list = {  # Your diseases list remains unchanged }

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', '')

    if not symptoms:
        return {"error": "Please provide symptoms"}, 400

    user_symptoms = [s.strip() for s in symptoms.split(',')]
    user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
    
    predicted_disease = get_predicted_value(user_symptoms)
    dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

    return {
        "predicted_disease": predicted_disease,
        "description": dis_des,
        "precautions": precautions,
        "medications": medications,
        "diet": rec_diet,
        "workout": workout
    }

# Additional routes
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

# Entry point
if __name__ == '__main__':
    app.run()

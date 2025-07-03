from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Replace this with the path to your trained model
MODEL_PATH = r"D:\Projects\CDC Course\Mental health\model.pkl"

# Load the model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/survey')
def survey():
    return render_template('survey.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        data = {
            'age': [int(request.form['age'])],
            'gender': [request.form['gender']],
            'country': [request.form['country']],
            'state': [request.form['state']],
            'self_employed': [request.form['self_employed']],
            'family_history': [request.form['family_history']],
            'work_interfere': [request.form['work_interfere']],
            'no_employees': [request.form['no_employees']],
            'remote_work': [request.form['remote_work']],
            'tech_company': [request.form['tech_company']],
            'benefits': [request.form['benefits']],
            'care_options': [request.form['care_options']],
            'wellness_program': [request.form['wellness_program']],
            'seek_help': [request.form['seek_help']],
            'anonymity': [request.form['anonymity']],
            'leave': [request.form['leave']],
            'mental_health_consequence': [request.form['mental_health_consequence']],
            'phys_health_consequence': [request.form['phys_health_consequence']],
            'coworkers': [request.form['coworkers']],
            'supervisor': [request.form['supervisor']],
            'mental_health_interview': [request.form['mental_health_interview']],
            'phys_health_interview': [request.form['phys_health_interview']],
            'mental_vs_physical': [request.form['mental_vs_physical']],
            'obs_consequence': [request.form['obs_consequence']]
        }

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Make prediction
        prediction = model.predict(df)[0]

        # Render result template with prediction
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
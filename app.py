import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
data = pd.read_csv('breast_cancer.csv')
pipe = pickle.load(open('breast_cancer.pkl','rb'))

@app.route("/")
def index():
    # locations = sorted(data['details'].unique())
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    radius_mean = float(request.form.get('radius_mean'))
    perimeter_mean = float(request.form.get('perimeter_mean')) 
    area_mean = float(request.form.get('area_mean'))
    compactness_mean = float(request.form.get('compactness_mean'))
    concave_points_mean = float(request.form.get('concave_points_mean'))
    radius_se = float(request.form.get('radius_se'))
    perimeter_se = float(request.form.get('perimeter_se'))
    area_se = float(request.form.get('area_se'))
    compactness_se= float(request.form.get('compactness_se'))
    concave_points_se= float(request.form.get('concave_points_se'))
    radius_worst= float(request.form.get('radius_worst'))
    perimeter_worst= float(request.form.get('perimeter_worst'))
    compactness_worst= float(request.form.get('compactness_worst'))
    concave_points_worst= float(request.form.get('concave_points_worst'))
    texture_worst= float(request.form.get('texture_worst'))
    area_worst= float(request.form.get('area_worst'))



    print(radius_mean, perimeter_mean, area_mean, compactness_mean,concave_points_mean,radius_se,perimeter_se,area_se,compactness_se,concave_points_se,radius_worst,perimeter_worst,compactness_worst,concave_points_worst,texture_worst,area_worst)
    input = np.array([[radius_mean, perimeter_mean, area_mean, compactness_mean,concave_points_mean,radius_se,perimeter_se,area_se,compactness_se,concave_points_se,radius_worst,perimeter_worst,compactness_worst,concave_points_worst,texture_worst,area_worst]])
    print(input)
   # prediction = pipe.predict(input)[0]
    proba = pipe.predict_proba(input)[0]
    print(proba)
    print(np.round(proba[1],5))
    return str(np.round(proba[1]*100,5))

if __name__ == '__main__':
    app.run(debug = True, host = "0.0.0.0", port = 9696)

 

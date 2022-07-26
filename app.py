# import relevant libraries for flask, htmlrendering and loading the ml model

from crypt import methods
from flask import Flask, request, url_for, redirect, render_template
import joblib
import pickle
import pandas as pd
from sklearn.preprocessing import scale

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
model = pickle.load(open("scale.pkl", "rb"))

@app.route("/")
def hello_world():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/predict", methods=['POST'])
def predict():
    pregnancies = request.form['1']
    glucose = request.form['2']
    bloodpressure = request.form['3']
    skinThickness = request.form['4']
    insulin = request.form['5']
    bmi = request.form['6']
    dpf = request.form['7']
    age = request.form['8']
    rowDF = pd.DataFrame([pd.Series([pregnancies, glucose, bloodpressure, skinThickness, insulin, bmi, dpf, age])])
    rowDF_new = pd.DataFrame(scale.transform(rowDF))
    print(rowDF_new)
    return render_template('index.html')
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
# Alle function aus =  preprocessin.py 
from preprocessing import encode_chest_pain_type, encode_thalassemia, apply_1_hot_encoding



# 1. Setup & Model laden
model = joblib.load('artifacts/svm_pipeline.pkl' )
st.set_page_config( 'Medicine Heart Diagnosed', ':book:', 'wide')


# 2. Titel
st.markdown("""
    <style>
    .main-title { 
        font-size: calc(20px + 2vw) !important; 
        color: orange !important; 
        font-family: 'Times New Roman', Times, serif !important; 
        text-align: center !important; 
        margin-bottom: 20px;
    }
     .title { 
        font-size: calc(8px + 1vw) !important; 
        color: orange !important; 
        font-family: 'Times New Roman', Times, serif !important; 
        text-align: center !important; 
        margin-bottom: 20px;
    }      
            
    </style>
    <h1 class="main-title">Medical heart diagnosis</h1>
            <p class="title">Fill it out and get the results instantly from our <br> machine learning system </p>
    """, unsafe_allow_html=True)


# 2.1 CREATE PLACEHOLDER AT THE TOP
chart_placeholder = st.empty()
message_placeholder = st.empty()

    
# 3. Daten laden und Input:
df= pd.read_csv('data/feature eng data/feature_eng_data.csv')

# 3.1 Eingabe-Maske (Slider und Selectboxen)
box_10, box_11, box_12, = st.columns(3)

age = box_10.slider('Age 29-77', min_value= df['age'].min(), max_value= df['age'].max())
sex = box_11.selectbox( 'Sex', options= df['sex'].unique())
chest_pain_type = box_12.selectbox( 'Chest pain type', options= df['chest_pain_type'].unique())


# 3.2 Eingabe-Maske (Slider und Selectboxen)
box_14, box_15, box_16 = st.columns(3)

resisting_blood_pressure = box_14.slider('Resisting blood pressure', min_value= df['resisting_blood_pressure'].min(), max_value= df['resisting_blood_pressure'].max())
cholesterol_level = box_15.slider('Cholesterol level', min_value= df["cholesterol_level"].min(), max_value= df["cholesterol_level"].max())
fasting_blood_sugar = box_16.selectbox('Fasting blood sugar', options= df["fasting_blood_sugar"].unique() ) 


# 3.3 Eingabe-Maske (Slider und Selectboxen)
box_17, box_18, box_19 = st.columns(3)

rest_ecg = box_17.selectbox('Rest ECG', options=df['rest_ecg'].unique())
max_heart_rate_achieved = box_18.selectbox( 'Max heart rate', options= df['max_heart_rate_achieved'].unique())
exercise_induced_angina = box_19.selectbox( 'Exercise induced angina', options= df['exercise_induced_angina'].unique())


# NEU: Du musst ALLE Felder abfragen, die das Modell braucht
box_20, box_21, box_22 , box_23 = st.columns(4)

st_depression = box_20.slider( 'depression', min_value= df["st_depression"].min(), max_value= df["st_depression"].max())
st_slope = box_21.selectbox('ST Slope', options=df['st_slope'].unique())
num_major_vessels = box_22.selectbox('Number of major vessels (0-3)', options=sorted(df['num_major_vessels'].unique()))
thalassemia = box_23.selectbox('Thalassemia', options=df['thalassemia'].unique())

# 4. Berechnung
# WICHTIG: Die Liste muss ALLE Spalten enthalten, die auch im Original-CSV sind
data = pd.DataFrame( 
    [[age, sex, chest_pain_type, resisting_blood_pressure,
       cholesterol_level, fasting_blood_sugar, rest_ecg,
       max_heart_rate_achieved, exercise_induced_angina, st_depression,
       st_slope, num_major_vessels, thalassemia]], 
    columns=['age', 'sex', 'chest_pain_type', 'resisting_blood_pressure',
             'cholesterol_level', 'fasting_blood_sugar', 'rest_ecg',
             'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression',
             'st_slope', 'num_major_vessels' , 'thalassemia']
)


# 5. Preprocessing
data = encode_chest_pain_type(data)
data = encode_thalassemia(data)


# ACHTUNG: 'diagnosis' muss aus der Liste in preprocessing.py entfernt werden, 
# da du die Diagnose ja erst vorhersagen willst und sie nicht als Input hast!
data = apply_1_hot_encoding(data, df)


# 6. Vorhersage
# Erzwinge die exakte Spaltenreihenfolge vom Training
data = data[model.feature_names_in_] 

# Sicherstellen, dass die Spaltenreihenfolge exakt wie im Training ist
prediction = model.predict(data)


# 6.1 Die rohe Vorhersage
raw_val = model.predict(data)[0]

# Manuelle Umwandlung in Text
if raw_val >= 0.5:
    result_text = "CRITICAL (Sick)"
    color = "red"
else:
    result_text = "STABLE (Healthy)"
    color = "green"

st.markdown(f"<h2 style='color:{color}'>{result_text} (Value: {raw_val:.3f})</h2>", unsafe_allow_html=True)


# 7. Vorhersage & Grafik
prediction = model.predict(data)
score = float(prediction) # Wandelt z.B. [0.473] in 0.473 um

# Das Diagramm erstellen
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = score,
    title = {'text': "Heart Disease Risk Score"},
    gauge = {
        'axis': {'range': [0, 1]},
        'steps': [
            {'range': [0, 0.5], 'color': "#2ecc71"}, # Green
            {'range': [0.5, 1], 'color': "#e74c3c"}  # Red
        ],
        'threshold': {'line': {'color': "black", 'width': 4}, 'value': 0.5}
    }
))

chart_placeholder.plotly_chart(fig, key="heart_risk_gauge")

# 6. UPDATE MESSAGE AT THE TOP
if score >= 0.5:
    message_placeholder.error(f"**High Risk Detected** (Score: {score:.2f}). Please consult a specialist.")
else:
    message_placeholder.success(f"**Low Risk** (Score: {score:.2f}). The patient's data appears stable.")